#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pickle
import datetime
import time
import numpy as np
import pandas as pd
from src import *


BASE_DIR = os.path.dirname(__file__)
logger = Logger('imputation_aa.{}.log'.format(datetime.datetime.now().strftime('%y%m%d%H%M')))
print('Logging to imputation_aa.log.')


def impute_aa(args):
    logger.log('Imputation processes started at {}.'.format(time.ctime()))

    # Load files
    print('Loading files...')
    dosage = pd.read_table(args.dosage + '.deephla.dosage', sep='\t|\s+', index_col=0, header=None, engine='python')
    with open(args.aa_table + '.aa_table.pickle', 'rb') as f:
        aa_table = pickle.load(f)
    out_prefix = args.out

    num_sample = dosage.shape[1]-2
    logger.log('{} people loaded from dosage file.'.format(num_sample))

    # Batch process
    batch_size = 1000
    num_batch = (num_sample-1)//batch_size+1

    aa_table_index = []
    hla_names = list(aa_table.keys())
    for hla in hla_names:
        aa_table_index.extend(list(aa_table[hla].columns))
    for batch_i in range(num_batch):
        logger.log('Batch process {}/{} started'.format(batch_i+1, num_batch))
        dosage_batch = dosage.iloc[:, 2+batch_size*batch_i:2+min(num_sample, batch_size*(batch_i+1))]
        num_sample_batch = dosage_batch.shape[1]
        aa_dosage_batch = pd.DataFrame(np.zeros((len(aa_table_index), num_sample_batch)), index=aa_table_index)
        for hla in hla_names:
            try:
                for i in aa_dosage_batch.columns:
                    aa_dosage_batch.loc[aa_table[hla].columns, i] = np.dot(dosage_batch.loc[aa_table[hla].index[2:]].iloc[:, i][dosage_batch.loc[aa_table[hla].index[2:]].iloc[:, i] != 0], \
                                        aa_table[hla].loc[dosage_batch.loc[aa_table[hla].index[2:]].iloc[:, i][dosage_batch.loc[aa_table[hla].index[2:]].iloc[:, i] != 0].index])
            except KeyError:
                logger.log('{} has no information of amino acid polymorphisms.'.format(hla))
                pass
        aa_dosage_batch.to_csv('{}.batch{}.aa.deephla.dosage'.format(out_prefix, batch_i+1), sep='\t', header=True, index=True)

    aa_dosage = pd.DataFrame(index=aa_table_index)
    aa_a1 = []
    aa_a2 = []
    for hla in hla_names:
        aa_a1.extend(list(aa_table[hla].loc['a1']))
        aa_a2.extend(list(aa_table[hla].loc['a2']))
    aa_dosage['a1'] = aa_a1
    aa_dosage['a2'] = aa_a2

    # Merge result files
    for batch_i in range(num_batch):
        aa_dosage_batch = pd.read_csv('{}.batch{}.aa.deephla.dosage'.format(out_prefix, batch_i+1), index_col=0, sep='\t')
        aa_dosage = pd.concat((aa_dosage, aa_dosage_batch), axis=1)
    aa_dosage.to_csv('{}.aa.deephla.dosage'.format(out_prefix), sep='\t', header=False, index=True)
    for batch_i in range(num_batch):
        os.remove('{}.batch{}.aa.deephla.dosage'.format(out_prefix, batch_i+1))

    logger.log('The processes have been finished at {}.'.format(time.ctime()))


def main():
    parser = argparse.ArgumentParser(description='Determine dosages of amino acid polymorphisms from 4-digit classical allele dosages.')
    parser.add_argument('--dosage', required=True, help='DEEP*HLA dosage file (.deephla.dosage).', dest='dosage')
    parser.add_argument('--aa-table', required=True, help='Correspondence table between HLA allele and amino acid polymorphism (.aa_table.pickle format).', dest='aa_table')
    parser.add_argument('--out', required=True, help='Prefix of result file', dest='out')

    args = parser.parse_args()

    impute_aa(args)


if __name__ == '__main__':
    main()
