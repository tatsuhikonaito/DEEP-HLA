#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
import datetime
import time
import tqdm
import numpy as np
import pandas as pd
import pickle
from scipy.stats import pearsonr
import torch
from torch.utils.data.dataset import Subset
from src import *


BASE_DIR = os.path.dirname(__file__)
CUDA_IS_AVAILABLE = torch.cuda.is_available()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logger = Logger('cross_validation.{0}.log'.format(datetime.datetime.now().strftime('%y%m%d%H%M')))
print('Logging to cross validation.log.')


def test_model(test_loader, hla_list, allele_cnts, num_task, model, metric, mode):
    torch.no_grad()
    test_acc = []

    for m in model:
        model[m] = model[m].cpu()
        model[m].eval()

    for batch in test_loader:
        shared_input = batch[0].requires_grad_(False)
        labels = {}
        for t in range(num_task):
            labels[t] = batch[t+1].requires_grad_(False)
        shared_output, mask_input, mask_1, mask_2 = model['shared'](shared_input.float(), None, None, None)
        for t in range(num_task):
            out_t, _ = model[t](shared_output.float(), None)
            metric[t].update(out_t, labels[t])

    t = 0
    for hla in hla_list:
        if not allele_cnts[hla] == 1:
            if mode == 'train':
                logger.log('{0} training accuracy: {1}'.format(hla, metric[t].get_result()['acc'].item()))
            elif mode == 'val':
                logger.log('{0} validation accuracy: {1}'.format(hla, metric[t].get_result()['acc'].item()))
            test_acc.append(metric[t].get_result()['acc'].item())
            metric[t].reset()
            t += 1
        else:
            if mode == 'best_val':
                test_acc.append('NA')

    return test_acc


def train_model(train_loader, hla_list, allele_cnts, num_task, model, optimizer, loss_fn, metric, epoch):
    if (epoch + 1) % 10 == 0:
        # Every 50 epoch, half the LR
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.85
        logger.log('Half the learning rate at epoch {0}.'.format(epoch))

    for m in model:
        if CUDA_IS_AVAILABLE:
            model[m] = model[m].cuda()
        model[m].train()

    with tqdm.tqdm(train_loader) as pbar:
        for i, batch in enumerate(pbar):
            shared_input = batch[0].requires_grad_(True)
            if CUDA_IS_AVAILABLE:
                shared_input = shared_input.cuda()

            labels = {}
            for t in range(num_task):
                labels[t] = batch[t+1]
                if CUDA_IS_AVAILABLE:
                    labels[t] = labels[t].cuda()

            # Scaling the loss functions
            loss_data = {}
            grads = {}
            scale = {}
            # Use mask variables as dropout in order to fix their values in the course of a epoch
            masks = {}
            mask_input = None
            mask1 = None
            mask2 = None

            optimizer.zero_grad()

            # First compute representations (z)
            with torch.no_grad():
                shared_output, mask_input, mask1, mask2 = model['shared'](shared_input.float(), mask_input, mask1, mask2)
            # As an approximate solution we only need gradients for input
            shared_variable = shared_output.clone().requires_grad_(True)

            # Compute gradients of each loss function wrt z
            for t in range(num_task):
                optimizer.zero_grad()
                out_t, masks[t] = model[t](shared_variable.float(), None)
                loss_t = loss_fn[t](out_t, labels[t].long())
                loss_data[t] = loss_t.data.item()
                loss_t.backward()
                grads[t] = []
                grads[t].append(shared_variable.grad.data.clone().requires_grad_(False))
                shared_variable.grad.data.zero_()

            if num_task != 1:
                # Normalize all gradients
                gn = gradient_normalizers(grads, loss_data)
                for t in range(num_task):
                    for gr_i in range(len(grads[t])):
                        grads[t][gr_i] = grads[t][gr_i] / gn[t]

                # Frank-Wolfe iteration to compute scales.
                sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in range(num_task)])
                for t in range(num_task):
                    scale[t] = float(sol[t])
            else:
                scale[0] = 1

            # Scaled back-propagation
            optimizer.zero_grad()
            shared_output, mask_input, mask1, mask2 = model['shared'](shared_input.float(), mask_input, mask1, mask2)
            for t in range(num_task):
                out_t, mask_t = model[t](shared_output.float(), masks[t])
                loss_t = loss_fn[t](out_t, labels[t].long())
                loss_data[t] = loss_t.data.item()
                if t > 0:
                    loss = loss + scale[t] * loss_t
                else:
                    loss = scale[t] * loss_t

            loss.backward()
            optimizer.step()

            pbar.set_description('[Epoch %d]' % epoch)

    train_acc = test_model(train_loader, hla_list, allele_cnts, num_task, model, metric, 'train')
    logger.log('Average training accuracy: {0}'.format(np.mean(train_acc)))

    torch.cuda.empty_cache()

    return model


def train(args):
    logger.log('Training processes started at {0}.'.format(time.ctime()))

    # Load files
    print('Loading files...')
    ref_bim = pd.read_table(args.ref + '.bim', sep='\t', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None)
    ref_phased = pd.read_table(args.ref + '.bgl.phased', sep='\t|\s+', header=None, engine='python', skiprows=5).iloc[:, 1:]
    ref_phased = ref_phased.set_index(1)
    ref_phased.columns = np.arange(ref_phased.shape[1])
    sample_bim = pd.read_table(args.sample + '.bim', sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')
    with open(args.model + '.model.json', 'r') as f:
        model_config = json.load(f)
    with open(args.hla + '.hla.json', 'r') as f:
        hla_info = json.load(f)
    model_dir = args.model_dir

    if args.max_digit == '2-digit':
        digit_list = ['2-digit']
    elif args.max_digit == '4-digit':
        digit_list = ['2-digit', '4-digit']
    elif args.max_digit == '6-digit':
        digit_list = ['2-digit', '4-digit', '6-digit']

    # Extract only SNPs which exist both in reference and sample data
    num_ref = ref_phased.shape[1] // 2
    logger.log('{0} people loaded from reference.'.format(num_ref))
    logger.log('{0} SNPs loaded from reference.'.format(len(ref_bim)))
    logger.log('{0} SNPs loaded from sample.'.format(len(sample_bim)))
    concord_snp = ref_bim.pos.isin(sample_bim.pos)
    for i in range(len(concord_snp)):
        if concord_snp.iloc[i]:
            tmp = np.where(sample_bim.pos == ref_bim.iloc[i].pos)[0][0]
            if set((ref_bim.iloc[i].a1, ref_bim.iloc[i].a2)) != \
                    set((sample_bim.iloc[tmp].a1, sample_bim.iloc[tmp].a2)):
                concord_snp.iloc[i] = False
    num_concord = np.sum(concord_snp)
    logger.log('{0} SNPs matched in position and used for training.'.format(num_concord))

    ref_concord_phased = ref_phased.iloc[np.where(concord_snp)[0]]
    model_bim = ref_bim.iloc[np.where(concord_snp)[0]]

    if not os.path.exists(os.path.join(BASE_DIR, model_dir)):
        os.mkdir(os.path.join(BASE_DIR, model_dir))
    else:
        print('Warning: Directory for saving models already exists.')

    model_bim.to_csv(os.path.join(BASE_DIR, model_dir, 'model.bim'), sep='\t', header=False, index=False)

    # Encode reference SNP data
    snp_encoded = np.zeros((2*num_ref, num_concord, 2))
    for i in range(num_concord):
        a1 = model_bim.iloc[i].a1
        a2 = model_bim.iloc[i].a2
        snp_encoded[ref_concord_phased.iloc[i, :] == a1, i, 0] = 1
        snp_encoded[ref_concord_phased.iloc[i, :] == a2, i, 1] = 1

    # Encode reference HLA data
    hla_encoded = {}
    for hla in hla_info:
        for i in range(2*num_ref):
            hla_encoded[hla] = {}
        for digit in digit_list:
            hla_encoded[hla][digit] = np.zeros(2 * num_ref)
            for j in range(len(hla_info[hla][digit])):
                hla_encoded[hla][digit][np.where(ref_phased.loc[hla_info[hla][digit][j]] == 'P')[0]] = j

    # Parameters for training
    val_split = args.val_split
    batch_size = 64
    num_epoch = args.num_epoch
    patience = args.patience

    # Generate CV result DataFrame
    result_index = []
    for v in hla_info.values():
        for digit in digit_list:
            result_index.extend(v[digit])
    cv_dosage = pd.DataFrame(index=result_index, columns=np.arange(num_ref))

    # Generate correct dosage
    try:
        correct_dosage = pd.read_csv(args.ref + '.deephla.dosage', header=None, index_col=0, sep='\t')
    except FileNotFoundError:
        correct_dosage = pd.DataFrame(np.zeros((len(result_index), num_ref)), index=result_index, columns=np.arange(num_ref))
        for hla in hla_info:
            for digit in digit_list:
                for j in range(len(hla_info[hla][digit])):
                    correct_dosage.loc[hla_info[hla][digit][j], np.where(ref_phased.iloc[:, np.arange(0, 2*num_ref, 2)].loc[hla_info[hla][digit][j]] == 'P')[0]] += 1
                    correct_dosage.loc[hla_info[hla][digit][j], np.where(ref_phased.iloc[:, np.arange(1, 2*num_ref+1, 2)].loc[hla_info[hla][digit][j]] == 'P')[0]] += 1
        correct_dosage["a1"] = "P"
        correct_dosage["a2"] = "A"
        correct_dosage = correct_dosage[["a1", "a2"] + list(correct_dosage.columns[:-2])]
        correct_dosage.to_csv(args.ref + '.deephla.dosage', header=False, index=True, sep='\t')

    # Cross_validation
    num_fold = args.num_fold
    for fold in range(num_fold):
        logger.log('Cross validation fold {0}/{1} started'.format(fold+1, num_fold))
        fold_model_dir = os.path.join(model_dir, 'fold{:02d}'.format(fold+1))
        if not os.path.exists(os.path.join(BASE_DIR, fold_model_dir)):
            os.mkdir(os.path.join(BASE_DIR, fold_model_dir))
        else:
            print('Warning: Directory for saving models of fold{:02d} already exists.'.format(fold))

        # Split indices into training, validation, and test
        tmp_index = np.concatenate((np.arange(2 * int(num_ref * (fold / num_fold))),
                                    np.arange(2 * int(num_ref * (fold + 1) / num_fold), 2 * num_ref)))
        test_index = np.arange(2 * int(num_ref * (fold / num_fold)), 2 * int(num_ref * (fold + 1) / num_fold))
        train_index = np.array([tmp_index[i] for i in np.arange(2 * int(num_ref * val_split), len(tmp_index))])
        val_index = np.array([tmp_index[i] for i in np.arange(2 * int(num_ref * val_split))])
        num_sample_split = int(len(test_index) / 2)

        result_best_val = pd.DataFrame(index=hla_info.keys(), columns=digit_list)
        result_best_test = pd.DataFrame(index=hla_info.keys(), columns=digit_list)
        for g in model_config:
            hla_list = model_config[g]['HLA']
            w = model_config[g]['w'] * 1000
            st = int(hla_info[hla_list[0]]['pos']) - w
            ed = int(hla_info[hla_list[-1]]['pos']) + w
            st_index = max(0, np.sum(model_bim.pos < st) - 1)
            ed_index = min(num_concord, num_concord - np.sum(model_bim.pos > ed))
            for digit in digit_list:
                logger.log('Training models for {0} at {1} level.'.format(', '.join(hla_list), digit))
                # Count HLA alleles
                allele_cnts = {}
                all_one_allele = True
                skip_hlas = []
                for hla in hla_list:
                    allele_cnts[hla] = len(hla_info[hla][digit])
                    if allele_cnts[hla] == 1:
                        skip_hlas.append(hla)
                    else:
                        all_one_allele = False
                if all_one_allele:
                    logger.log('Skipped group {0} at {1} level because all genes have only one allele.'.format(g, digit))
                    continue
                elif len(skip_hlas) != 0:
                    logger.log('Skipped {0} at {1} level because of only one allele.'.format(', '.join(skip_hlas), digit))
    
                # Generate training data
                train_data = []
                for i in range(2*num_ref):
                    tmp = [snp_encoded[i, st_index:ed_index]]
                    for hla in hla_list:
                        if not allele_cnts[hla] == 1:
                            tmp.append(hla_encoded[hla][digit][i])
                    train_data.append(tmp)
                num_task = len(train_data[0]) - 1
    
                # Spare the latter part of data for validation
                train_loader = torch.utils.data.DataLoader(Subset(train_data, train_index), batch_size=batch_size)
                val_loader = torch.utils.data.DataLoader(Subset(train_data, val_index), batch_size=batch_size, shuffle=False)
                test_loader = torch.utils.data.DataLoader(Subset(train_data, test_index), batch_size=batch_size, shuffle=False)
    
                # Generate models
                model = {'shared': SharedNet(model_config[g], ed_index-st_index, input_collapse=False)}
                # Transfer parameters of shared net from those of upper digit
                if not digit == '2-digit':
                    try:
                        model['shared'].load_state_dict(torch.load(os.path.join(BASE_DIR, model_dir, '{0}_{1}_shared_model.pickle'.format(g, digit_list[digit_list.index(digit) - 1]))))
                        logger.log('Transferred parameters from shared net of upper digit at {0} level.'.format(digit))
                    except FileNotFoundError:
                        logger.log('Shared net of upper digit not found at {0} level.'.format(digit))
                        pass
                t = 0
                for hla in hla_list:
                    if not allele_cnts[hla] == 1:
                        model[t] = EachNet(model_config[g], allele_cnts[hla])
                        t += 1
                for m in model:
                    model[m] = model[m].float()
                    model[m].train()
                model_params = []
                for m in model:
                    model_params += model[m].parameters()
    
                optimizer = torch.optim.Adam(model_params)
                loss_fn = get_loss(num_task)
                metric = get_metrics(num_task)
    
                best_epoch = 1
                best_ave_val_acc = 0
                patience_cnt = 0
    
                # Training iteration
                for epoch in range(1, num_epoch+1):
                    model = train_model(train_loader, hla_list, allele_cnts, num_task, model, optimizer, loss_fn, metric, epoch)
                    val_acc = test_model(val_loader, hla_list, allele_cnts, num_task, model, metric, 'val')
                    ave_val_acc = np.mean(val_acc)
                    logger.log('Average validation accuracy: {0}'.format(ave_val_acc))
                    # Save the current model if the current model is equal or better than the best one
                    if ave_val_acc >= best_ave_val_acc:
                        torch.save(model['shared'].state_dict(), os.path.join(BASE_DIR, fold_model_dir, '{0}_{1}_epoch{2}_shared_model.pickle'.format(g, digit, epoch)))
                        t = 0
                        for hla in hla_list:
                            if not allele_cnts[hla] == 1:
                                torch.save(model[t].state_dict(), os.path.join(BASE_DIR, fold_model_dir, '{0}_{1}_epoch{2}_{3}_model.pickle'.format(g, digit, epoch, hla)))
                                t += 1
                        if not epoch == 1:
                            os.remove(os.path.join(BASE_DIR, fold_model_dir, '{0}_{1}_epoch{2}_shared_model.pickle'.format(g, digit, best_epoch)))
                            for hla in hla_list:
                                if not allele_cnts[hla] == 1:
                                    os.remove(os.path.join(BASE_DIR, fold_model_dir, '{0}_{1}_epoch{2}_{3}_model.pickle'.format(g, digit, best_epoch, hla)))
                        best_epoch = epoch
                        if ave_val_acc > best_ave_val_acc:
                            best_ave_val_acc = ave_val_acc
                            patience_cnt = 0
                    # Increase patience count if the current model is not better than the best one
                    if ave_val_acc <= best_ave_val_acc:
                        patience_cnt += 1
                        # Early stopping when patience count reaches to the upper limit
                        if patience_cnt >= patience:
                            logger.log('Early stopping.')
                            break
                    if epoch == num_epoch:
                        logger.log('All epochs finished without early stopping.')
                logger.log('The best model is at epoch {0}.'.format(best_epoch))
    
                # Rename models
                os.rename(os.path.join(BASE_DIR, fold_model_dir, '{0}_{1}_epoch{2}_shared_model.pickle'.format(g, digit, best_epoch)), 
                          os.path.join(BASE_DIR, fold_model_dir, '{0}_{1}_shared_model.pickle'.format(g, digit)))
                for hla in hla_list:
                    if not allele_cnts[hla] == 1:
                        os.rename(os.path.join(BASE_DIR, fold_model_dir, '{0}_{1}_epoch{2}_{3}_model.pickle'.format(g, digit, best_epoch, hla)), 
                                  os.path.join(BASE_DIR, fold_model_dir, '{0}_{1}_{2}_model.pickle'.format(g, digit, hla)))
                best_val_acc = test_model(val_loader, hla_list, allele_cnts, num_task, model, metric, 'best_val')
                result_best_val.loc[hla_list, digit] = best_val_acc
                best_test_acc = test_model(test_loader, hla_list, allele_cnts, num_task, model, metric, 'best_val')
                result_best_test.loc[hla_list, digit] = best_test_acc

                # Load the best models
                model['shared'].load_state_dict(
                    torch.load(os.path.join(BASE_DIR, fold_model_dir, '{0}_{1}_shared_model.pickle'.format(g, digit))))
                t = 0
                for hla in hla_list:
                    if not allele_cnts[hla] == 1:
                        model[t].load_state_dict(
                            torch.load(os.path.join(BASE_DIR, fold_model_dir, '{0}_{1}_{2}_model.pickle'.format(g, digit, hla))))
                        t += 1
                for m in model:
                    model[m] = model[m].float()
                    model[m].eval()
                torch.no_grad()

                shared, mask_input, mask_1, mask_2 = model['shared'](torch.Tensor(snp_encoded[test_index, st_index:ed_index]).float(), None, None, None)
                t = 0
                for hla in hla_list:
                    if allele_cnts[hla] == 1:
                        dosage = np.full((1, num_sample_split), 2.000)
                    else:
                        imputed, mask_fc = model[t](shared, None)
                        imputed = np.array(np.exp(imputed.detach()).T)
                        dosage = np.array([[imputed[j, 2 * i] + imputed[j, 2 * i + 1] for i in range(num_sample_split)] for j in
                                           range(imputed.shape[0])])
                        t += 1
                    cv_dosage.loc[hla_info[hla][digit], int(num_ref * (fold/num_fold)):int(num_ref * (fold+1)/num_fold)-1] = dosage

        result_best_val.to_csv(os.path.join(BASE_DIR, fold_model_dir, 'best_val.txt'), header=True, index=True, sep='\t')
        result_best_test.to_csv(os.path.join(BASE_DIR, fold_model_dir, 'best_test.txt'), header=True, index=True, sep='\t')

    cv_dosage["a1"] = "P"
    cv_dosage["a2"] = "A"
    cv_dosage = cv_dosage[["a1", "a2"] + list(cv_dosage.columns[:-2])]
    cv_dosage.to_csv(os.path.join(BASE_DIR, model_dir, 'cv.deephla.dosage'), header=False, index=True, sep='\t')

    # Calculate correlation coeffients between correct and CV-imputed dosages.
    accuracy = pd.DataFrame(index=result_index, columns=["r2"])
    accuracy.index.name = "Allele"
    for i in accuracy.index:
        try:
            accuracy.loc[i, "r2"] = pearsonr(cv_dosage.loc[i][2:], correct_dosage.loc[i][2:])[0]**2
        except ValueError:
            accuracy.loc[i, "r2"] = "NA"
    accuracy.to_csv(os.path.join(BASE_DIR, model_dir, 'CV_accuracy.txt'), 
                    index=True, header=True, sep='\t')

    print('The processes have been finished at {0}.'.format(time.ctime()))


def main():
    parser = argparse.ArgumentParser(description='Perform cross-validation using a HLA reference data.')
    parser.add_argument('--ref', required=True, help='HLA reference data (.bgl.phased or .haps, and .bim format).', dest='ref')
    parser.add_argument('--sample', required=True, help='Sample SNP data (.bim format).', dest='sample')
    parser.add_argument('--model', required=True, help='Model configuration (.model.json format).', dest='model')
    parser.add_argument('--hla', required=True, help='HLA information of the reference data (.hla.json format).', dest='hla')
    parser.add_argument('--model-dir', default='model', required=False, help='Directory for saving trained models.', dest='model_dir')
    parser.add_argument('--num-epoch', default=100, type=int, required=False, help='Number of epochs to train.', dest='num_epoch')
    parser.add_argument('--patience', default=16, type=int, required=False, help='Patience for early-stopping.', dest='patience')
    parser.add_argument('--val-split', default=0.05, type=float, required=False, help='Ratio of splitting data for validation.', dest='val_split')
    parser.add_argument('--max-digit', default='6-digit', choices=['2-digit', '4-digit', '6-digit'], required=False, help='Maximum resolution of classical alleles to impute.', dest='max_digit')
    parser.add_argument('--num-fold', default=10, type=int, required=False, help='Number of folds in cross-validation.', dest='num_fold')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
