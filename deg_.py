'''
Bernstein degree
Author: Zhenhuan(Neyo) Yang
'''

import os
import numpy as np
import multiprocessing as mp
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle
from sklearn.model_selection import RepeatedKFold
from sauc_ import SAUC
from get_idx import get_idx

def single_run(para):

    '''
    for multiprocessing mapping function with variable
    input:
        para -
    output:
    '''

    # unfold parameters
    i,train_index,test_index,m = para
    n_tr = len(train_index)
    options['ids'] = get_idx(n_tr, options['n_pass'])

    # X and y must be global here to avoid multiprocessing sharing
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Define model parameter
    options['m'] = m

    # implement algorithm
    elapsed_time, roc_auc = SAUC(X_train, X_test, y_train, y_test, options)

    return i,m,roc_auc


def cv(num_cpus, n_splits, n_repeats, M):

    '''
    Cross validation by multiprocessing
    input:
        num_cpus -
        M -
    '''

    # record auc
    ROC_AUC = pd.DataFrame()

    # record parameters
    input_paras = []

    # cross validation prepare
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=7)
    for i, (train_index, test_index) in enumerate(rkf.split(X)):
        for m in M:
            input_paras.append((i, train_index, test_index, m))

    # cross validation run on multiprocessors
    with mp.Pool(processes=num_cpus) as pool:
        results_pool = pool.map(single_run, input_paras)
        pool.close()
        pool.join()

    # get results
    for i, m, roc_auc in results_pool:
        ROC_AUC[(i,m)] = roc_auc

    return ROC_AUC

if __name__ == '__main__':

    # Define what to run this time
    datasets = ['sector.scale']
    names = ['hinge']
    num_cpus = 15
    n_splits = 3
    n_repeats = 1

    options = {}
    options['n_pass'] = 1
    options['rec'] = .5
    options['name'] = 'hinge'

    # Define model parameter
    options['R'] = .01
    options['c'] = .1


    # Define model parameter to search
    M = [2,5,10,25,40]


    for dataset in datasets:

        print('Loading dataset = %s ......' % (dataset), end=' ')
        X, y = load_svmlight_file('/home/neyo/PycharmProjects/AUC/bi-datasets/%s' % (dataset))
        X = X.toarray()
        X, y = shuffle(X, y, random_state=10)
        print('Done!')


        for name in names:

            if os.path.isfile('/home/neyo/PycharmProjects/AUC/results/deg_%s_%s.h5' % (name, dataset)):

                df = pd.read_pickle('/home/neyo/PycharmProjects/AUC/results/deg_%s_%s.h5' % (name, dataset))
                result = df.to_dict()
            else:
                result = {}

            roc_auc = cv(num_cpus, n_splits, n_repeats, M)

            for m in M:

                if m in result.keys():
                    pass
                else:
                    result[(m)] = {}

                for i in range(n_splits * n_repeats):
                    result[(m)][i] = roc_auc[(i, m)]

            # Results
            df = pd.DataFrame(result)
            df.to_pickle('/home/neyo/PycharmProjects/AUC/results/deg_%s_%s.h5' % (name,dataset))