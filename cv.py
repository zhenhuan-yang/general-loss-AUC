'''
Cross validation

Author: Zhenhuan(Neyo) Yang
'''

import numpy as np
import multiprocessing as mp
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
    folder, alg, trte = para
    training, testing = trte

    # FEATURES and LABELS must be global here to avoid multiprocessing sharing
    Xtr = FEATURES[training]
    Ytr = LABELS[training]
    Xte = FEATURES[testing]
    Yte = LABELS[testing]

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

    return folder,max(roc_auc)


def cv(alg, n, folders, num_cpus):

    '''
    Cross validation by multiprocessing

    input:
        alg - algorithm
        n - number of samples
        folders - number of folders
        num_cpus -
    '''

    # record auc
    ROC_AUC = np.zeros(folders)

    # record parameters
    input_paras = []

    # cross validation prepare
    for folder in range(folders):
        training, testing = split(n, folder, folders)
        trte = training, testing
        input_paras.append((folder,alg,trte))

    # cross validation run on multiprocessors
    with mp.Pool(processes=num_cpus) as pool:
        results_pool = pool.map(single_run, input_paras)
        pool.close()
        pool.join()

    # get results
    for folder, roc_auc in results_pool:
        ROC_AUC[folder] = roc_auc

    mean = np.mean(ROC_AUC)
    std = np.std(ROC_AUC)

    return mean, std


if __name__ == '__main__':

    # Define hyper parameters
    options = {}
    options['T'] = 100

    # Define model parameter
    options['L'] = 2
    options['c'] = 1

    # Define what to run this time
    dataset = 'a9a'
    alg = 'SAUC'
    folders = 3
    num_cpus = 15

    # Read data from hdf5 file
    hf = h5py.File('/home/neyo/PycharmProjects/AUC/datasets/%s.h5' % (dataset), 'r')
    FEATURES = hf['FEATURES'][:]
    LABELS = hf['LABELS'][:]
    hf.close()

    m = len(LABELS)

    # Run
    N = [1,10,20,30,40,50,60,70,80,90,100]
    NAME = ['hinge','logistic']
    res = {}
    for name in NAME:
        options['name'] = name
        res[name] = {}
        res[name]['upper'] = []
        res[name]['middle'] = []
        res[name]['lower'] = []
        for n in N:
            options['N'] = n
            mean, std = cv(alg, m, folders, num_cpus)
            res[name]['upper'].append(mean+std)
            res[name]['middle'].append(mean)
            res[name]['lower'].append(mean-std)

    # Plot N vs AUC
    KEY = ['upper','middle','lower']
    COLOR = ['r','b']
    fig = plt.figure(figsize=(12, 4))  # create a figure object
    fig.suptitle(dataset)
    for name,color in zip(NAME,COLOR):
        for key in KEY:
            if key == 'middle':
                plt.plot(N,res[name][key],'-o',color=color,label=name)
            else:
                plt.plot(N,res[name][key],'--o',color=color)
    plt.xlabel('Degree of Bernstein Polynomial(m)')
    plt.ylabel('AUC')
    plt.legend(loc=4)
    plt.show()
