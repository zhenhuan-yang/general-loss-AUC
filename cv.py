'''
Cross validation

Author: Zhenhuan(Neyo) Yang
'''

import numpy as np
import multiprocessing as mp
from itertools import product
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
from sklearn.utils import shuffle
import h5py
import matplotlib.pyplot as plt
from SAUC import SAUC
from OAM import OAM
from SPAM import SPAM
from OPAUC import OPAUC
from SOLAM import SOLAM
from FSAUC import FSAUC
from split import split

def single_run(para):

    '''
    for multiprocessing mapping function with variable

    input:
        para -

    output:

    '''

    # unfold parameters
    folder, alg, trte,c,r = para
    training, testing = trte

    # FEATURES and LABELS must be global here to avoid multiprocessing sharing
    Xtr = X[training]
    Ytr = y[training]
    Xte = X[testing]
    Yte = y[testing]

    # Define model parameter
    options['c'] = c
    options['R'] = r

    # implement algorithm
    if alg =='SAUC':
        elapsed_time, roc_auc = SAUC(Xtr, Xte, Ytr, Yte, options)
    elif alg == 'OAM':
        elapsed_time, roc_auc = OAM(Xtr, Xte, Ytr, Yte, options)
    elif alg == 'SOLAM':
        elapsed_time, roc_auc = SOLAM(Xtr, Xte, Ytr, Yte, options)
    elif alg == 'FSAUC':
        elapsed_time, roc_auc = FSAUC(Xtr, Xte, Ytr, Yte, options)
    elif alg == 'OPAUC':
        elapsed_time, roc_auc = OPAUC(Xtr, Xte, Ytr, Yte, options)
    elif alg == 'SPAM':
        elapsed_time, roc_auc = SPAM(Xtr, Xte, Ytr, Yte, options)
    else:
        print('Wrong algorithm!')
        elapsed_time = []
        roc_auc = []

    return folder,c,r,roc_auc


def cv(alg, n, folders, num_cpus, C, R):

    '''
    Cross validation by multiprocessing

    input:
        alg - algorithm
        n - number of samples
        folders - number of folders
        num_cpus -
        C -
        R -
    '''

    # record auc
    ROC_AUC = pd.DataFrame()

    # record parameters
    input_paras = []

    # cross validation prepare
    for folder in range(folders):
        training, testing = split(n, folder, folders)
        trte = training, testing
        for c,r in product(C,R):
            input_paras.append((folder,alg,trte,c,r))

    # cross validation run on multiprocessors
    with mp.Pool(processes=num_cpus) as pool:
        results_pool = pool.map(single_run, input_paras)
        pool.close()
        pool.join()

    # get results
    for folder, c, r, roc_auc in results_pool:
        ROC_AUC[(folder,c,r)] = roc_auc

    return ROC_AUC


if __name__ == '__main__':

    # Define what to run this time
    datasets = ['skin_nonskin']
    algs = ['SAUC']
    folders = 3
    num_cpus = 15

    # Define hyper parameters
    options = {}
    options['name'] = 'hinge'
    options['T'] = 10
    options['N'] = 10
    options['delta'] = .1
    options['option'] = 'gradient'
    options['sampling'] = 'reservoir'

    # Define model parameter
    options['Np'] = 100
    options['Nn'] = 100
    options['tau'] = 50

    # Define model parameter to search
    # R = [2**i for i in range(-2,-1)] + [3**i for i in range(-2,-1)] + [5**i for i in range(-2,4)] + [10**i for i in range(-2,3)]
    R = [10 ** i for i in range(-1, 0)]
    # C = [5**i for i in range(-3,-1)] + [10**i for i in range(-2,1)]
    C = [10 ** i for i in range(-2, 3)]

    for dataset in datasets:

        X, y = load_svmlight_file('/home/neyo/PycharmProjects/AUC/bi-datasets/%s' % (dataset), dtype = np.float32)

        X = X.toarray()

        X, y = shuffle(X, y, random_state= 7)

        m = len(y)

        for alg in algs:
            ROC_AUC = cv(alg, m, folders, num_cpus, C, R)
            result = {}
            # Results
            for c, r in product(C, R):

                result[(c, r)] = {}
                do = []
                for folder in range(folders):
                    result[(c,r)][folder] = ROC_AUC[(folder, c, r)]
                    do.append(max(result[c,r][folder]))

                MEAN = np.mean(do)
                STD = np.std(do)

                print('alg = %s data = %s c = %.2f R = %.2f AUC = ' % (alg, dataset, c, r), end=' ')
                print(('%.4f$\pm$' % MEAN).lstrip('0'), end='')
                print(('%.4f' % STD).lstrip('0'))

            # Results
            df = pd.DataFrame(result)

            df.to_pickle('/home/neyo/PycharmProjects/AUC/results/cv_%s_%s.h5' % (alg, dataset))
