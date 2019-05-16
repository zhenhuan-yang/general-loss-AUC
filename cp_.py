'''
Convergence plot
Author: Zhenhuan(Neyo) Yang
'''

import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from solam_ import SOLAM
from sauc_ import SAUC
from spam_ import SPAM
from oam_ import OAM
from opauc_ import OPAUC
from fsauc_ import FSAUC
from get_idx import get_idx

if __name__ == '__main__':

    # Define hyper parameters
    options = {}
    options['name'] = 'hinge'
    options['n_pass'] = 2
    options['rec'] = 0.5  # record when

    # Define model parameter
    options['m'] = 5
    options['R'] = 1
    options['c'] = 1
    options['Np'] = 100
    options['Nn'] = 100
    options['delta'] = .1
    options['tau'] = 50


    # Define what to run this time
    dataset = 'cod-rna'
    ALG = ['SPAM']

    print('Loading dataset = %s ......' %(dataset), end=' ')
    # hf = h5py.File('/home/neyo/PycharmProjects/AUC/h5-datasets/%s.h5' % (dataset), 'r')
    # X = hf['FEATURES'][:]
    # y = hf['LABELS'][:]
    # hf.close()

    X, y = load_svmlight_file('/home/neyo/PycharmProjects/AUC/bi-datasets/%s' %(dataset))
    X = X.toarray()
    # Simple prepare training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=7)
    n_tr = len(y_train)
    options['ids'] = get_idx(n_tr, options['n_pass'])
    print('Done!')

    # Prepare results
    res = {}
    for alg in ALG:
        if alg == 'SOLAM':
            res[alg] = SOLAM(X_train, X_test, y_train, y_test, options)
        elif alg == 'FSAUC':
            res[alg] = FSAUC(X_train, X_test, y_train, y_test, options)
        elif alg == 'OAM':
            res[alg] = OAM(X_train, X_test, y_train, y_test, options)
        elif alg == 'OPAUC':
            res[alg] = OPAUC(X_train, X_test, y_train, y_test, options)
        elif alg == 'SPAM':
            res[alg] = SPAM(X_train, X_test, y_train, y_test, options)
        elif alg == 'SAUC':
            res[alg] = SAUC(X_train, X_test, y_train, y_test, options)
        else:
            print('Wrong Algorithm!')

    # Plot results
    fig = plt.figure()  # create a figure object
    fig.suptitle(dataset)
    for alg in ALG:
        plt.plot(res[alg][0],res[alg][1],label = alg)
    plt.xlabel('CPU Time(s)')
    plt.ylabel('AUC')
    plt.ylim([0.5,1])
    plt.legend(loc=4)
    plt.show()

    fig.savefig('/home/neyo/PycharmProjects/AUC/results/cp_%s.png' % (dataset))