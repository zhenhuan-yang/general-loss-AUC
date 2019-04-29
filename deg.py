'''
Bernstein degree
Author: Zhenhuan(Neyo) Yang
'''

import h5py
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import pandas as pd
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
    Xtr = FEATURES[training]
    Ytr = LABELS[training]
    Xte = FEATURES[testing]
    Yte = LABELS[testing]

    # Define model parameter
    options['N'] = m

    # implement algorithm
    elapsed_time, roc_auc = SAUC(Xtr, Ytr, Xte, Yte, options)

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
    datasets = ['a9a']
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
    N = [5,10,20]

    for dataset in datasets:

        # Read data from hdf5 file
        hf = h5py.File('/home/neyo/PycharmProjects/AUC/h5-datasets/%s.h5' % (dataset), 'r')
        FEATURES = hf['FEATURES'][:]
        LABELS = hf['LABELS'][:]
        hf.close()

        n = len(LABELS)

        # Run
        ROC_AUC = cv(n, folders, num_cpus, N)

        result = {}
        line = []
        error = []
        # Results
        for m in N:
            ROC = np.zeros((folders, options['T']))

            for folder in range(folders):
                ROC[folder] = ROC_AUC[(folder, m)]

            result[(m)]['MEAN'] = np.mean(ROC, axis=0)
            result[(m)]['STD'] = np.std(ROC, axis=0)

            ind = np.max(result[(m)]['MEAN'])
            line.append(result[(m)]['MEAN'][ind])
            error.append(result[(m)]['STD'][ind])

        plt.errorbar(N,line,yerr=error,fmt='-o')

        # Results
        df = pd.DataFrame(result)

        df.to_pickle(r'C:\Users\zy572688\PycharmProjects\AUC\results\deg_%s_%s.h5' % (dataset))
