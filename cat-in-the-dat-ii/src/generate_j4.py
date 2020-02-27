#!/usr/bin/env python
from __future__ import division
import argparse
import os
import logging
import time

import numpy as np
import pandas as pd
from scipy import stats, sparse
from kaggler.preprocessing import LabelEncoder, TargetEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from kaggler.data_io import load_data, save_data

from const import ID_COL, TARGET_COL, N_FOLD, SEED


def generate_feature(train_file, test_file, train_feature_file,
                     test_feature_file, feature_map_file):
    logging.info('loading raw data')
    trn = pd.read_csv(train_file, index_col=ID_COL)
    tst = pd.read_csv(test_file, index_col=ID_COL)

    logging.info('label encoding categorical variables')

    y = trn.loc[:, TARGET_COL]
    n_trn = trn.shape[0]
    trn = trn.drop(TARGET_COL, axis=1)
    df = pd.concat([trn, tst], axis=0)

    # build features
    features_bin = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']
    features_cat = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']
    features_hex = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
    features_ord = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']
    features_cyc = ['day', 'month']

    logging.info("Dummy encode: bin 0 to 4")
    # convert bins 0, 1, 2 to object so that
    # get_dummies recognizes them and creates missing indicators
    bin_012 = ['bin_0', 'bin_1', 'bin_2']
    df[bin_012] = df[bin_012].astype(object)

    dummies = pd.get_dummies(df[features_bin], dummy_na=True)
    df = df.drop(features_bin, axis=1)
    df = pd.concat([df, dummies], axis=1)

    logging.info("Target encoding: nom 0 to 9 and cyclical features")
    target_enc_cols = features_ord + features_cat + features_hex + features_cyc
    cv = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
    te = TargetEncoder(cv=cv)
    te.fit(trn.loc[:, target_enc_cols], y)
    df.loc[:, target_enc_cols] = te.transform(df.loc[:, target_enc_cols])

#    logging.info("Label encode ordinals: ord 0 to 5")
#    map_ord_0 = None  # already a numeric column
#    map_ord_1 = {'Novice': 1, 'Contributor': 2,
#                 'Expert': 3, 'Master': 4, 'Grandmaster': 5}
#    map_ord_2 = {'Freezing': 1, 'Cold': 2, 'Warm': 3,
#                 'Hot': 4, 'Boiling Hot': 5, 'Lava Hot': 6}
#    map_ord_3 = dict(zip(df['ord_3'].value_counts().sort_index().keys(),
#                         range(1, len(df['ord_3'].value_counts()) + 1)))
#    map_ord_4 = dict(zip(df['ord_4'].value_counts().sort_index().keys(),
#                         range(1, len(df['ord_4'].value_counts()) + 1)))
#
#    temp_ord_5 = pd.DataFrame(
#        df['ord_5'].value_counts().sort_index().keys(), columns=['ord_5'])
#    temp_ord_5['First'] = temp_ord_5['ord_5'].astype(str).str[0].str.upper()
#    temp_ord_5['Second'] = temp_ord_5['ord_5'].astype(str).str[1].str.upper()
#    temp_ord_5['First'] = temp_ord_5['First'].replace(map_ord_4)
#    temp_ord_5['Second'] = temp_ord_5['Second'].replace(map_ord_4)
#    temp_ord_5['Add'] = temp_ord_5['First'] + temp_ord_5['Second']
#    temp_ord_5['Mul'] = temp_ord_5['First'] * temp_ord_5['Second']
#    map_ord_5 = dict(zip(temp_ord_5['ord_5'],
#                         temp_ord_5['Mul']))
#
#    maps = [map_ord_0, map_ord_1, map_ord_2, map_ord_3, map_ord_4, map_ord_5]
#    for i, m in zip(range(0, 6), maps):
#        if i != 0:
#            df[f'ord_{i}'] = df[f'ord_{i}'].map(m)
#        df[f'ord_{i}'] = (df[f'ord_{i}'].fillna(df[f'ord_{i}'].median()))

#    logging.info("cyclical features")
#    df[features_cyc] = df[features_cyc].astype(object)
#    dummies_cyc = pd.get_dummies(df[features_cyc], dummy_na=True)
#    df = df.drop(features_cyc, axis=1)
#    df = pd.concat([df, dummies_cyc], axis=1)

    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(df.columns):
            f.write('{}\t{}\tq\n'.format(i, col))

    logging.info('saving features')
    save_data(df.values[:n_trn, ], y.values, train_feature_file)

    save_data(df.values[n_trn:, ], None, test_feature_file)


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-feature-file',
                        required=True, dest='train_feature_file')
    parser.add_argument('--test-feature-file',
                        required=True, dest='test_feature_file')
    parser.add_argument('--feature-map-file', required=True,
                        dest='feature_map_file')

    args = parser.parse_args()

    start = time.time()
    generate_feature(args.train_file,
                     args.test_file,
                     args.train_feature_file,
                     args.test_feature_file,
                     args.feature_map_file)
    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))
