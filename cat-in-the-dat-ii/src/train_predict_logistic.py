#!/usr/bin/env python

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score as AUC

import argparse
import logging
import numpy as np
import operator
import os
import pandas as pd
import time

from const import N_FOLD, SEED
from kaggler.data_io import load_data
from sklearn import linear_model


def train_predict(train_file, test_file, predict_valid_file, 
		  predict_test_file, retrain=True):

    model_name = os.path.splitext(os.path.splitext(os.path.basename(predict_test_file))[0])[0]

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='{}.log'.format(model_name))

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)

    logging.info('Loading CV Ids')
    cv = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

    p = np.zeros(X.shape[0])
    p_tst = np.zeros(X_tst.shape[0])
    n_bests = []
    for i, (i_trn, i_val) in enumerate(cv.split(X, y), 1):
        logging.info('Training model #{}'.format(i))
        glm = linear_model.LogisticRegression(solver='lbfgs',
					      max_iter=2020,
					      fit_intercept=True,
					      penalty='none',
					      verbose=0)
        glm.fit(X[i_trn], y[i_trn])
        p[i_val] = glm.predict_proba(X[i_val])[:, 1]
        logging.info('CV #{}: {:.4f}'.format(i, AUC(y[i_val], p[i_val])))

        if not retrain:
            p_tst += glm.predict_proba(X_tst)[:,1] / N_FOLD

    logging.info('CV: {:.4f}'.format(AUC(y, p)))
    logging.info('Saving validation predictions...')
    np.savetxt(predict_valid_file, p, fmt='%.6f', delimiter=',')

    if retrain:
        logging.info('Retraining with 100% training data')
        
        glm = linear_model.LogisticRegression(random_state=1,
					      solver='lbfgs',
					      max_iter=2020,
					      fit_intercept=True,
					      penalty='none',
					      verbose=0)
        glb = glb.fit(X, y)
        p_tst = glb.predict_proba(X_tst)

    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')
    parser.add_argument('--retrain', default=False, action='store_true')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  retrain=args.retrain)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
