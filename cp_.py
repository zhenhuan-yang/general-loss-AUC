'''
Convergence plot
Author: Zhenhuan(Neyo) Yang
'''

import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sauc_ import SAUC
from spam_ import SPAM
from fsauc_ import FSAUC
from opauc_ import OPAUC
from oam_ import OAM
from solam_ import SOLAM
from get_idx import get_idx

if __name__ == '__main__':

    # Define hyper parameters
    options = {}
    options['name'] = 'hinge'
    options['rec'] = 0.5  # record when

    # Define model parameter
    options['m'] = 10
    options['Np'] = 100
    options['Nn'] = 100
    options['delta'] = .1
    options['tau'] = 50


    # Define what to run this time
    dataset = 'news20'
    ALG = ['SAUC']

    print('Loading dataset = %s ......' %(dataset), end=' ')
    X, y = load_svmlight_file('/home/neyo/PycharmProjects/AUC/bi-datasets/%s' %(dataset))
    X = X.toarray()
    # Simple prepare training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.33, random_state=10)
    n_tr = len(y_train)

    print('Done!')

    # Prepare results
    for alg in ALG:
        res = {}
        res['elapsed_time'] = []
        res['roc_auc'] = []
        if alg == 'OAM':
            options['R'] = 10
            options['c'] = 1
            options['n_pass'] = 1
            options['ids'] = get_idx(n_tr, options['n_pass'])
            res['elapsed_time'],res['roc_auc'] = OAM(X_train, X_test, y_train, y_test, options)
            df = pd.DataFrame(res)
            df.to_pickle('/home/neyo/PycharmProjects/AUC/results/cp_%s_%s.h5' % (alg,dataset))
            print('Done!')
        elif alg == 'SAUC':
            options['R'] = .01
            options['c'] = 1
            options['n_pass'] = 5
            options['ids'] = get_idx(n_tr, options['n_pass'])
            res['elapsed_time'], res['roc_auc'] = SAUC(X_train, X_test, y_train, y_test, options)
            df = pd.DataFrame(res)
            df.to_pickle('/home/neyo/PycharmProjects/AUC/results/cp_%s_%s.h5' % (alg,dataset))
            print('Done!')
        elif alg == 'SOLAM':
            options['R'] = 100
            options['c'] = .1
            options['n_pass'] = 2
            options['ids'] = get_idx(n_tr,options['n_pass'])
            res['elapsed_time'], res['roc_auc'] = SOLAM(X_train, X_test, y_train, y_test, options)
            print('Done!')
        elif alg == 'SPAM':
            options['R'] = 1
            options['c'] = .1
            options['n_pass'] = 2
            options['ids'] = get_idx(n_tr,options['n_pass'])
            res['elapsed_time'], res['roc_auc'] = SPAM(X_train, X_test, y_train, y_test, options)
            print('Done!')
        elif alg == 'FSAUC':
            options['R'] = 100
            options['c'] = .1
            options['n_pass'] = 2
            options['ids'] = get_idx(n_tr,options['n_pass'])
            res['elapsed_time'], res['roc_auc'] = FSAUC(X_train, X_test, y_train, y_test, options)
            print('Done!')
        elif alg == 'OPAUC':
            options['R'] = 100
            options['c'] = .1
            options['n_pass'] = 2
            options['ids'] = get_idx(n_tr,options['n_pass'])
            res['elapsed_time'], res['roc_auc'] = OPAUC(X_train, X_test, y_train, y_test, options)
            print('Done!')
        else:
            print('Wrong Algorithm!')


