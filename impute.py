#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import csv
import json
import datetime
import time
import numpy as np
import pandas as pd
from scipy.stats import entropy
import torch
from src import *


BASE_DIR = os.path.dirname(__file__)
CUDA_IS_AVAILABLE = torch.cuda.is_available()
os.environ['KMP_DUPLICATE_LIB_OK']='True'
logger = Logger('imputation.{}.log'.format(datetime.datetime.now().strftime('%y%m%d%H%M')))
print('Logging to imputation.log.')


def impute(args):
    logger.log('Imputation processes started at {}.'.format(time.ctime()))

    # Load files
    print('Loading files...')
    with open(args.model + '.model.json', 'r') as f:
        model_config = json.load(f)
    sample_bim = pd.read_table(args.sample + '.bim', sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')
    sample_fam = pd.read_table(args.sample + '.fam', sep='\t|\s+', names=['fid', 'iid', 'fat', 'mot', 'sex', 'phe'], header=None, engine='python')
    model_dir = args.model_dir
    model_bim = pd.read_table(os.path.join(model_dir, 'model.bim'), sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')
    with open(args.hla + '.hla.json', 'r') as f:
        hla_info = json.load(f)
    if args.max_digit == '2-digit':
        digit_list = ['2-digit']
    elif args.max_digit == '4-digit':
        digit_list = ['2-digit', '4-digit']
    elif args.max_digit == '6-digit':
        digit_list = ['2-digit', '4-digit', '6-digit']
    mc_dropout = args.mc_dropout
    out_prefix = args.out

    num_sample = len(sample_fam)
    num_snp = len(model_bim)
    num_sample_snp = len(sample_bim)
    concord_snp = model_bim.pos.isin(sample_bim.pos)
    concord_sample_snp = sample_bim.pos.isin(model_bim.pos)
    dup_sample_snp = sample_bim.loc[sample_bim.pos.isin(model_bim.pos), "pos"].duplicated()     # Added Jan 1st 2024

    logger.log('{} SNPs used for training.'.format(num_snp))
    logger.log('{} people loaded from sample.'.format(num_sample))
    logger.log('{} SNPs loaded from sample.'.format(num_sample_snp))
    logger.log('{} SNPs consistent with model SNPs in position and used for imputation.'.format(np.sum(concord_snp)))

    # Load and encode sample phased data
    sample_encoded = np.zeros((2 * num_sample, num_snp, 2))
    if args.phased_type == 'haps':
        if os.path.exists(args.sample + '.haps.gz'):
            import gzip
            f_phased = gzip.open(args.sample + '.haps.gz', mode="rt")
        else:
            f_phased = open(args.sample + '.haps', mode="r")
        reader = csv.reader(f_phased, delimiter=' ')
        for i in range(num_sample_snp):
            line = next(reader)
            if concord_sample_snp[i]:
                if dup_sample_snp[i]:     # Added Jan 1st 2024
                    continue
                a1 = model_bim[model_bim.pos == sample_bim.iloc[i].pos].a1.iloc[0]
                a2 = model_bim[model_bim.pos == sample_bim.iloc[i].pos].a2.iloc[0]
                #i_ = np.where(model_bim.pos == sample_bim.iloc[i].pos)[0]
                i_ = np.where(model_bim.pos == sample_bim.iloc[i].pos)[0][0]     # Added Jan 1st 2024
                if a1 == line[3] and a2 == line[4]:
                    sample_encoded[np.where(np.array(line[5:]) == '0')[0], i_, 0] = 1
                    sample_encoded[np.where(np.array(line[5:]) == '1')[0], i_, 1] = 1
                elif a1 == line[4] and a2 == line[3]:
                    sample_encoded[np.where(np.array(line[5:]) == '1')[0], i_, 0] = 1
                    sample_encoded[np.where(np.array(line[5:]) == '0')[0], i_, 1] = 1
                else:
                    logger.log('Warning: Bases are not consistent at position {} between model and sample'.format(model_bim.iloc[i].pos))
    elif args.phased_type == 'bgl':
        sample_phased = pd.read_table(args.sample + '.bgl.phased', sep='\t|\s+', header=None, engine='python', skiprows=5).iloc[:, 1:]
        sample_phased = sample_phased.set_index(1)

    # Batch process
    batch_size = 1000
    num_batch = (2*num_sample-1)//batch_size+1
    result_index = []
    for v in hla_info.values():
        for digit in digit_list:
            result_index.extend(v[digit])
    for batch_i in range(num_batch):
        logger.log('Batch process {}/{} started'.format(batch_i + 1, num_batch))
        sample_fam_batch = sample_fam.iloc[
                           batch_size * batch_i//2:min(2*num_sample, batch_size*(batch_i+1))//2]
        num_sample_batch = len(sample_fam_batch)
        result_phased = pd.DataFrame(index=result_index,
                                     columns=[sample_fam_batch.iloc[i//2].iid for i in range(2*num_sample_batch)])
        result_dosage = pd.DataFrame(index=result_index, columns=sample_fam_batch.iid)
        sample_encoded_batch = sample_encoded[batch_size*batch_i: min(2*num_sample, batch_size*(batch_i+1))]
        if mc_dropout:
            result_entropy = pd.DataFrame(index=result_index,
                                          columns=[sample_fam_batch.iloc[i//2].iid for i in range(2*num_sample_batch)])

        for g in model_config:
            hla_list = model_config[g]['HLA']
            w = model_config[g]['w'] * 1000
            st = int(hla_info[hla_list[0]]['pos']) - w
            ed = int(hla_info[hla_list[-1]]['pos']) + w
            st_index = max(0, np.sum(model_bim.pos < st) - 1)
            ed_index = min(num_snp, num_snp - np.sum(model_bim.pos > ed))
            sample_input_batch = sample_encoded_batch[:, st_index:ed_index, :]

            for digit in digit_list:
                logger.log('Imputing {} at {} level.'.format(', '.join(hla_list), digit))
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
                    logger.log('Skipped group {} at {} level because all genes have only one allele.'.format(g, digit))
                    for hla in hla_list:
                        result_phased.loc[hla_info[hla][digit], sample_fam_batch.iid] = np.full((1, 2*num_sample_batch), 'P')
                        result_dosage.loc[hla_info[hla][digit], sample_fam_batch.iid] = np.full((1, num_sample_batch), 2.000)
                        if mc_dropout:
                            result_entropy.loc[hla_info[hla][digit], sample_fam_batch.iid] = 0
                    continue
                elif len(skip_hlas) != 0:
                    logger.log('Skipped {} at {} level because of only one allele.'.format(', '.join(skip_hlas), digit))

                # Load models
                model = {'shared': SharedNet(model_config[g], ed_index-st_index, input_collapse=False)}
                model['shared'].load_state_dict(torch.load(os.path.join(model_dir, '{}_{}_shared_model.pickle'.format(g, digit))))
                for hla in hla_list:
                    if not allele_cnts[hla] == 1:
                        model[hla] = EachNet(model_config[g], allele_cnts[hla])
                        model[hla].load_state_dict(torch.load(os.path.join(model_dir, '{}_{}_{}_model.pickle'.format(g, digit, hla))))
                for m in model:
                    model[m] = model[m].float()
                    model[m].eval()
                torch.no_grad()

                shared, mask_input, mask_1, mask_2 = model['shared'](torch.Tensor(sample_input_batch).float(), None, None, None)
                for hla in hla_list:
                    if allele_cnts[hla] == 1:
                        phased = np.full((1, 2*num_sample_batch), 'P')
                        dosage = np.full((1, num_sample_batch), 2.000)
                    else:
                        imputed, mask_fc = model[hla](shared, None)
                        imputed = np.array(np.exp(imputed.detach()).T)
                        phased = np.full((allele_cnts[hla], 2*num_sample_batch), 'A')
                        for i in range(2*num_sample_batch):
                            phased[np.argmax(imputed[:, i]), i] = 'P'
                        dosage = np.array([[imputed[j, 2*i]+imputed[j, 2*i+1] for i in range(num_sample_batch)] for j in range(imputed.shape[0])])
                    result_phased.loc[hla_info[hla][digit], sample_fam_batch.iid] = phased
                    result_dosage.loc[hla_info[hla][digit], sample_fam_batch.iid] = dosage

                # Monte Carlo dropout sampling
                if mc_dropout:
                    num_sampling = 2
                    logger.log('Sampling with dropout {} times to calculate uncertainty of prediction'.format(num_sampling))
                    prediction_stack = {}
                    for hla in hla_list:
                        if not allele_cnts[hla] == 1:
                            prediction_stack[hla] = np.zeros((2*num_sample, allele_cnts[hla]))
                    for m in model:
                        model[m].train()
                    for t in range(num_sampling):
                        shared, mask_input, mask_1, mask_2 = model['shared'](torch.Tensor(sample_input_batch).float(), None, None, None)
                        for hla in hla_list:
                            if not allele_cnts[hla] == 1:
                                imputed, mask_fc = model[hla](shared, None)
                                imputed = np.array(imputed.detach().T)
                                for i in range(2*num_sample_batch):
                                    prediction_stack[hla][i, np.argmax(imputed[:, i])] += 1
                    # Calculate entropy for each allele of each sample
                    for hla in hla_list:
                        if not allele_cnts[hla] == 1:
                            for i in range(allele_cnts[hla]):
                                tmp = np.zeros(2*num_sample_batch)
                                for j in range(2*num_sample_batch):
                                    tmp[j] = entropy([prediction_stack[hla][j][i], num_sampling-prediction_stack[hla][j][i]])
                                result_entropy.loc[hla_info[hla][digit][i], sample_fam_batch.iid] = tmp
                        else:
                            result_entropy.loc[hla_info[hla][digit], sample_fam_batch.iid] = 0
            result_phased.to_csv('{}.batch{}.deephla.phased'.format(out_prefix, batch_i+1), header=True, index=True, sep='\t')
            result_dosage.astype(np.float64).round(3).to_csv('{}.batch{}.deephla.dosage'.format(out_prefix, batch_i+1), header=True, index=True, sep='\t')
            if mc_dropout:
                result_entropy.to_csv('{}.batch{}.deephla.entropy'.format(out_prefix, batch_i+1), header=True, index=True, sep='\t')

    # Merge result files
    result_phased = pd.DataFrame()
    result_dosage = pd.DataFrame([['P', 'A'] for _ in range(len(result_index))], index=result_index)
    for batch_i in range(num_batch):
        result_phased_batch = pd.read_csv('{}.batch{}.deephla.phased'.format(out_prefix, batch_i+1), index_col=0, sep='\t')
        result_dosage_batch = pd.read_csv('{}.batch{}.deephla.dosage'.format(out_prefix, batch_i+1), index_col=0, sep='\t')
        result_phased = pd.concat((result_phased, result_phased_batch), axis=1)
        result_dosage = pd.concat((result_dosage, result_dosage_batch), axis=1)
    result_phased.to_csv('{}.deephla.phased'.format(out_prefix), header=False, index=True, sep='\t')
    result_dosage.iloc[:, 2:] = result_dosage.iloc[:, 2:].astype(np.float64).round(3)
    result_dosage.to_csv('{}.deephla.dosage'.format(out_prefix), header=False, index=True, sep='\t')
    for batch_i in range(num_batch):
        os.remove('{}.batch{}.deephla.phased'.format(out_prefix, batch_i+1))
        os.remove('{}.batch{}.deephla.dosage'.format(out_prefix, batch_i+1))
    if mc_dropout:
        result_entropy = pd.DataFrame()
        for batch_i in range(num_batch):
            result_entropy_batch = pd.read_csv('{}.batch{}.deephla.entropy'.format(out_prefix, batch_i+1), index_col=0, sep='\t')
            result_entropy = pd.concat((result_entropy, result_entropy_batch), axis=1)
        result_entropy.to_csv('{}.deephla.entropy'.format(out_prefix), header=False, index=True, sep='\t')
        for batch_i in range(num_batch):
            os.remove('{}.batch{}.deephla.entropy'.format(out_prefix, batch_i+1))

    logger.log('The processes have been finished at {}.'.format(time.ctime()))


def main():
    parser = argparse.ArgumentParser(description='Perform HLA imputation with a trained model.')
    parser.add_argument('--sample', required=True, help='Sample SNP data (.bgl.phased or .haps, and .bim format).', dest='sample')
    parser.add_argument('--model', required=True, help='Model configuration (.model.json format).', dest='model')
    parser.add_argument('--phased-type', default='bgl', choices=['bgl', 'haps'], required=False, help='File format of sample phased file ("bgl", "haps").', dest='phased_type')
    parser.add_argument('--hla', required=True, help='HLA information of the reference data (.hla.json format).', dest='hla')
    parser.add_argument('--model-dir', default=os.path.join(BASE_DIR, 'model'), required=False, help='Directory for saving trained models.', dest='model_dir')
    parser.add_argument('--out', required=True, help='Prefix of result file', dest='out')
    parser.add_argument('--max-digit', default='4-digit', choices=['2-digit', '4-digit', '6-digit'], required=False, help='Maximum resolution of alleles to impute.', dest='max_digit')
    parser.add_argument('--mc-dropout', default=False, type=bool, choices=[True, False], required=False, help='Whether to calculate uncertainty by Monte Carlo dropout.', dest='mc_dropout')

    args = parser.parse_args()

    impute(args)


if __name__ == '__main__':
    main()
