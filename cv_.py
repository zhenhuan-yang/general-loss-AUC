'''
Cross validation
Author: Zhenhuan(Neyo) Yang
'''

import numpy as np
import multiprocessing as mp
from itertools import product
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle
from sklearn.model_selection import RepeatedKFold
from sauc_ import SAUC
from oam_ import OAM
from spam_ import SPAM
from opauc_ import OPAUC
from solam_ import SOLAM
from fsauc_ import FSAUC
from get_idx import get_idx

def single_run(para):

    '''
    for multiprocessing mapping function with variable
    input:
        para -
    output:
    '''

    # unfold parameters
    alg,i,train_index,test_index,c,r = para
    n_tr = len(train_index)
    options['ids'] = get_idx(n_tr, options['n_pass'])

    # X and y must be global here to avoid multiprocessing sharing
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Define model parameter
    options['c'] = c
    options['R'] = r

    # implement algorithm
    if alg =='SAUC':
        elapsed_time, roc_auc = SAUC(X_train, X_test, y_train, y_test, options)
    elif alg == 'OAM':
        elapsed_time, roc_auc = OAM(X_train, X_test, y_train, y_test, options)
    elif alg == 'SOLAM':
        elapsed_time, roc_auc = SOLAM(X_train, X_test, y_train, y_test, options)
    elif alg == 'FSAUC':
        elapsed_time, roc_auc = FSAUC(X_train, X_test, y_train, y_test, options)
    elif alg == 'OPAUC':
        elapsed_time, roc_auc = OPAUC(X_train, X_test, y_train, y_test, options)
    elif alg == 'SPAM':
        elapsed_time, roc_auc = SPAM(X_train, X_test, y_train, y_test, options)
    else:
        print('Wrong algorithm!')
        elapsed_time = []
        roc_auc = []

    return i,c,r,roc_auc


def cv(alg, num_cpus, n_splits, n_repeats, C, R):

    '''
    Cross validation by multiprocessing
    input:
        alg - algorithm
        train_index -
        test_index -
        num_cpus -
        C -
        R -
    '''

    # record auc
    ROC_AUC = pd.DataFrame()

    # record parameters
    input_paras = []

    # cross validation prepare
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=7)
    for i,(train_index, test_index) in enumerate(rkf.split(X)):
        for c,r in product(C,R):
            input_paras.append((alg,i,train_index,test_index,c,r))

    # cross validation run on multiprocessors
    with mp.Pool(processes=num_cpus) as pool:
        results_pool = pool.map(single_run, input_paras)
        pool.close()
        pool.join()

    # get results
    for i, c, r, roc_auc in results_pool:
        ROC_AUC[(i,c,r)] = roc_auc

    return ROC_AUC


if __name__ == '__main__':

    # Define what to run this time
    datasets = ['smallNORB']
    algs = ['SOLAM']
    num_cpus = 15
    n_splits = 3
    n_repeats = 1

    # Define hyper parameters
    options = {}
    options['name'] = 'hinge'
    options['m'] = 1
    options['delta'] = .1
    options['tau'] = 50
    options['n_pass'] = 3
    options['rec'] = .5

    # Define model parameter
    options['Np'] = 100
    options['Nn'] = 100

    # Define model parameter to search
    # R = [2**i for i in range(-2,-1)] + [3**i for i in range(-2,-1)] + [5**i for i in range(-2,-1)]
    R = [10**i for i in range(1,2)]
    # C = [2**i for i in range(-3,4)] + [3**i for i in range(-3,4)] + [5**i for i in range(-2,3)]
    C = [10**i for i in range(1,2)]
    for dataset in datasets:

        print('Loading dataset = %s ......' %(dataset), end=' ')
        X, y = load_svmlight_file('/home/neyo/PycharmProjects/AUC/bi-datasets/%s' % (dataset))
        X = X.toarray()
        X, y = shuffle(X, y, random_state = 10)
        print('Done!')

        for alg in algs:

            # Run
            roc_auc = cv(alg, num_cpus, n_splits, n_repeats, C, R)
            result = {}

            # Results
            for c, r in product(C, R):
                result[(c, r)] = {}
                # find minimal recorded length
                L = np.zeros(n_splits * n_repeats)
                for i in range(n_splits * n_repeats):
                    L[i] = len(roc_auc[(i,c,r)])
                l = int(min(L))
                cv_roc_auc = np.zeros((n_splits * n_repeats, l))
                for i in range(n_splits * n_repeats):
                    cv_roc_auc[i] = roc_auc[(i, c, r)][:l]

                result[(c, r)]['MEAN'] = np.mean(cv_roc_auc, axis=0)
                result[(c, r)]['STD'] = np.std(cv_roc_auc, axis=0)

                print('alg = %s data = %s c = %.2f R = %.2f AUC = ' % (alg, dataset, c, r), end=' ')
                print(('%.4f$\pm$' % max(result[(c,r)]['MEAN'])).lstrip('0'), end='')
                print(('%.4f' % min(result[(c,r)]['STD'])).lstrip('0'))

            # Results
            df = pd.DataFrame(result)

            df.to_pickle('/home/neyo/PycharmProjects/AUC/results/cv_%s_%s.h5' % (alg, dataset))