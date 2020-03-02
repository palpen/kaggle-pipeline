#!/usr/bin/env python
from __future__ import division
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time

from sklearn.model_selection import StratifiedKFold
from kaggler.data_io import load_data, save_data
from kaggler.preprocessing import LabelEncoder, TargetEncoder

from const import ID_COL, TARGET_COL, SEED, N_FOLD


def generate_feature(train_file, test_file, train_feature_file,
                     test_feature_file, feature_map_file):
    logging.info('loading raw data')
    trn = pd.read_csv(train_file, index_col=ID_COL)
    tst = pd.read_csv(test_file, index_col=ID_COL)

    y = trn[TARGET_COL]
    trn.drop(TARGET_COL, axis=1, inplace=True)

    trn['time_feat'] = trn.reset_index()['time']
    tst['time_feat'] = tst.reset_index()['time']

    # Generate feature identifying batches
    # batches are 500,000 records long
    # This will be used to generate grouped by lagged features
#    batch_size= 500000
#    n_trn = trn.shape[0]
#    for i in range(1,int((n_trn / batch_size) + 1)):
#	beg = (i-1)*batch_size
#	end = i*batch_size -1
#	trn.loc[beg:end, f'batch'] = i
#
#    n_tst = tst.shape[0]
#    for i in range(1,int((n_tst / batch_size) + 1)):
#	beg = (i-1)*batch_size
#	end = i*batch_size -1
#	tst.loc[beg:end, f'batch'] = i

    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(trn.columns):
            f.write('{}\t{}\tq\n'.format(i, col))

    logging.info('saving features')
    save_data(trn.values, y.values, train_feature_file)
    save_data(tst.values, None, test_feature_file)


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-feature-file', required=True, dest='train_feature_file')
    parser.add_argument('--test-feature-file', required=True, dest='test_feature_file')
    parser.add_argument('--feature-map-file', required=True, dest='feature_map_file')

    args = parser.parse_args()

    start = time.time()
    generate_feature(args.train_file,
                     args.test_file,
                     args.train_feature_file,
                     args.test_feature_file,
                     args.feature_map_file)
    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))

