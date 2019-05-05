'''
Bernstein degree
Author: Zhenhuan(Neyo) Yang
'''

import h5py
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
from sklearn.utils import shuffle
from SAUC import SAUC
from split import split

def single_run(para):

    '''
    for multiprocessing mapping function with variable
    input:
        para -
    output:
    '''

    # unfold parameters
    folder, trte, m = para
    training, testing = trte

    # FEATURES and LABELS must be global here to avoid multiprocessing sharing
    Xtr = X[training]
    Ytr = y[training]
    Xte = X[testing]
    Yte = y[testing]

    # Define model parameter
    options['N'] = m

    # implement algorithm
    elapsed_time, roc_auc = SAUC(Xtr, Xte, Ytr, Yte, options)

    return folder,m,roc_auc


def cv(n, folders, num_cpus, N):

    '''
    Cross validation by multiprocessing
    input:
        n - number of samples
        folders - number of folders
        num_cpus -
        N -
    '''

    # record auc
    ROC_AUC = pd.DataFrame()

    # record parameters
    input_paras = []

    # cross validation prepare
    for folder in range(folders):
        training, testing = split(n, folder, folders)
        trte = training, testing
        for m in N:
            input_paras.append((folder,trte,m))

    # cross validation run on multiprocessors
    with mp.Pool(processes=num_cpus) as pool:
        results_pool = pool.map(single_run, input_paras)
        pool.close()
        pool.join()

    # get results
    for folder, m, roc_auc in results_pool:
        ROC_AUC[(folder,m)] = roc_auc

    return ROC_AUC

if __name__ == '__main__':

    # Define what to run this time
    datasets = ['cod-rna']
    folders = 3
    num_cpus = 15

    # Define hyper parameters
    options = {}
    options['name'] = 'hinge'
    options['T'] = 100

    # Define model parameter
    options['R'] = 1
    options['c'] = 1

    # Define Bernstein degree
    N = [1,5,10,25,50]

    for dataset in datasets:

        X, y = load_svmlight_file('/home/neyo/PycharmProjects/AUC/datasets/%s' % (dataset))

        X = preprocessing.normalize(X)

        X = X.toarray()

        X, y = shuffle(X, y, random_state=7)

        n = len(y)

        # Run
        ROC_AUC = cv(n, folders, num_cpus, N)

        result = {}
        line = []
        error = []
        # Results
        for m in N:
            ROC = np.zeros((folders, options['T']))
            result[(m)] = {}
            for folder in range(folders):
                ROC[folder] = ROC_AUC[(folder, m)]

            result[(m)]['MEAN'] = np.mean(ROC, axis=0)
            result[(m)]['STD'] = np.std(ROC, axis=0)

        # Results
        df = pd.DataFrame(result)

        df.to_pickle('/home/neyo/PycharmProjects/AUC/results/deg_%s.h5' % (dataset))