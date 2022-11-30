#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from collections import OrderedDict
import json
import pandas as pd


BASE_DIR = os.path.dirname(__file__)

# Specify the first 3 characters of HLA genes typed in the reference panel
prefix_hla = ['HLA', 'MIC', 'TAP']

def make_hlainfo(args):

    # Load files
    print('Loading files...')
    ref_bim = pd.read_table(args.ref + '.bim', sep='\t', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None)

    print('Making an hla.json file for {0}.'.format(args.ref))

    if args.max_digit == '2-digit':
        digit_list = ['2-digit']
    elif args.max_digit == '4-digit':
        digit_list = ['2-digit', '4-digit']
    elif args.max_digit == '6-digit':
        digit_list = ['2-digit', '4-digit', '6-digit']

    allele_list_all = [i for i in ref_bim.id if i[0:3] in prefix_hla]
    
    # Read a field separator of HLA alleles
    if len([i for i in allele_list_all if ':' in i]) != 0:
        sep = ':'
    else:
        sep = ''

    # Get HLA gene names
    # Remove shared epitopes
    hla_list = ['_'.join(i.split('_')[0:-1]) for i in allele_list_all if not i.split('_')[-2]=='SE']
    hla_list = sorted(set(hla_list), key=hla_list.index)

    # Make an hla information file
    hla_info = OrderedDict()
    for hla in hla_list:
        hla_info[hla] = OrderedDict()
        allele_list = [i for i in allele_list_all if i[0:len(hla)] == hla and i[len(hla)] == '_']
        hla_info[hla]['pos'] = str(ref_bim[ref_bim.id==allele_list[0]].pos.values[0])
    
        if sep == ':':
            # For the separator ':', judge the digit of each allele by counting ':'
            for d_i in range(len(digit_list)):
                digit = digit_list[d_i]
                hla_info[hla][digit] = [i for i in allele_list if i.count(':') == d_i]
        else:
            # For the separator '', judge the digit of each allele based on the length of the allele name
            # We assume that the first field might be 2- or 3-characters-long
            for d_i in range(len(digit_list)):
                digit = digit_list[d_i]
                hla_info[hla][digit] = [i for i in allele_list if len(i.split('_')[-1])==2*(d_i+1) or len(i.split('_')[-1])==2*(d_i+1)+1]

            if len(hla_info[hla][digit]) == 0:
                print('Warning: {0} alleles of {1} are not typed.'.format(digit, hla))

    if args.output:
        output_fname = args.output
    else:
        output_fname = os.path.join(BASE_DIR, args.ref + '.hla.json')

    with open(output_fname, 'w') as f:
        json.dump(hla_info, f)

    print('Done.')


def main():
    parser = argparse.ArgumentParser(description='Make an HLA information file.')
    parser.add_argument('--ref', required=True, 
                        help='HLA reference data (.bgl.phased or .haps, and .bim format).', dest='ref')
    parser.add_argument('--max-digit', default='4-digit', choices=['2-digit', '4-digit', '6-digit'], required=False,
                        help='Maximum resolution of classical alleles typed in the reference panel.', dest='max_digit')
    parser.add_argument('--output', required=False, help='Output filename for HLA information JSON file.', dest='output')

    args = parser.parse_args()

    make_hlainfo(args)


if __name__ == '__main__':
    main()
