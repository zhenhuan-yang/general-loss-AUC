'''
Cross validation

Author: Zhenhuan(Neyo) Yang
'''

import numpy as np
import multiprocessing as mp
from itertools import product
import pandas as pd
import h5py
from math import fabs
import matplotlib.pyplot as plt
from SAUC import SAUC
from OAM import OAM
from SPAM import SPAM
from OPAUC import OPAUC
from SOLAM import SOLAM
from FSAUC import FSAUC

def split(n, folder, folders):

    '''
    Split training and testing

    input:
        n - number of samples
        folder - number as testing folder
        folders - number of folders

    output:
        train_list -
        test_list -
    '''

    if folder >= folders:
        print('Exceed maximum folders!')
        return

    # regular portion of each folder
    portion = round(n / folders)
    start = portion * folder
    stop = portion * (folder + 1)

    if folders == 1:
        train_list = [i for i in range(n)]
        test_list = [i for i in range(n)]

    elif folders == 2:
        if folder == 0:
            train_list = [i for i in range(start)] + [i for i in range(stop, n)]
            test_list = [i for i in range(start, stop)]
        else:
            train_list = [i for i in range(start)]
            test_list = [i for i in range(start, n)]

    else:
        if fabs(stop - n) < portion:  # remainder occurs
            train_list = [i for i in range(start)]
            test_list = [i for i in range(start, n)]
        else:
            train_list = [i for i in range(start)] + [i for i in range(stop, n)]
            test_list = [i for i in range(start, stop)]

    return train_list, test_list

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
    Xtr = FEATURES[training]
    Ytr = LABELS[training]
    Xte = FEATURES[testing]
    Yte = LABELS[testing]

    # Define model parameter
    options['c'] = c
    options['R'] = r

    # implement algorithm
    if alg =='SAUC':
        elapsed_time, roc_auc = SAUC(Xtr, Ytr, Xte, Yte, options)
    elif alg == 'OAM':
        elapsed_time, roc_auc = OAM(Xtr, Ytr, Xte, Yte, options)
    elif alg == 'SOLAM':
        elapsed_time, roc_auc = SOLAM(Xtr, Ytr, Xte, Yte, options)
    elif alg == 'FSAUC':
        elapsed_time, roc_auc = FSAUC(Xtr, Ytr, Xte, Yte, options)
    elif alg == 'OPAUC':
        elapsed_time, roc_auc = OPAUC(Xtr, Ytr, Xte, Yte, options)
    elif alg == 'SPAM':
        elapsed_time, roc_auc = SPAM(Xtr, Ytr, Xte, Yte, options)
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
    dataset = 'a9a'
    alg = 'SAUC'
    folders = 2
    num_cpus = 15

    # Read data from hdf5 file
    hf = h5py.File('/home/neyo/PycharmProjects/AUC/h5-datasets/%s.h5' % (dataset), 'r')
    FEATURES = hf['FEATURES'][:]
    LABELS = hf['LABELS'][:]
    hf.close()

    m = len(LABELS)

    # Define hyper parameters
    options = {}
    options['name'] = 'hinge'
    options['T'] = 100
    options['N'] = 5
    options['option'] = 'sequential'
    options['sampling'] = 'reservoir'


    # Define model parameter
    options['Np'] = 100
    options['Nn'] = 100
    options['B'] = 200

    # Define model parameter to search
    R = [.1,1,10,100]
    C = [.1,1,10,100]

    # Run
    ROC_AUC = cv(alg,m,folders,num_cpus,C,R)

    # Results
    ROC_AUC.to_pickle('/home/neyo/PycharmProjects/AUC/results/cv_%s_%s.h5' % (alg,dataset))



