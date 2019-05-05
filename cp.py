'''
Convergence plot
Author: Zhenhuan(Neyo) Yang
'''

import h5py
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from SOLAM import SOLAM
from SAUC import SAUC
from SPAM import SPAM
from OAM import OAM
from OPAUC import OPAUC
from FSAUC import FSAUC

if __name__ == '__main__':

    # Define hyper parameters
    options = {}
    options['name'] = 'hinge'
    options['option'] = 'gradient'
    options['sampling'] = 'reservoir'

    # Define model parameter
    options['N'] = 5
    options['R'] = 1
    options['c'] = 1
    options['Np'] = 100
    options['Nn'] = 100
    options['delta'] = .1
    options['tau'] = 50


    # Define what to run this time
    dataset = 'real-sim'
    ALG = ['OAM']

    print('Loading dataset = %s ......' %(dataset), end=' ')
    # hf = h5py.File('/home/neyo/PycharmProjects/AUC/h5-datasets/%s.h5' % (dataset), 'r')
    # X = hf['FEATURES'][:]
    # y = hf['LABELS'][:]
    # hf.close()

    X, y = load_svmlight_file('/home/neyo/PycharmProjects/AUC/datasets/%s' %(dataset))

    X = preprocessing.normalize(X)

    X = X.toarray()

    # Simple prepare training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

    print('Done!')

    # Prepare results
    res = {}
    for alg in ALG:
        if alg == 'SOLAM':
            options['T'] = 2000
            res[alg] = SOLAM(X_train, X_test, y_train, y_test, options)
        elif alg == 'FSAUC':
            options['T'] = 500
            res[alg] = FSAUC(X_train, X_test, y_train, y_test, options)
        elif alg == 'OAM':
            options['T'] = 2000
            res[alg] = OAM(X_train, X_test, y_train, y_test, options)
        elif alg == 'OPAUC':
            options['T'] = 3000
            res[alg] = OPAUC(X_train, X_test, y_train, y_test, options)
        elif alg == 'SPAM':
            options['T'] = 3000
            res[alg] = SPAM(X_train, X_test, y_train, y_test, options)
        elif alg == 'SAUC':
            options['T'] = 200
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
    plt.ylim([.5,1])
    plt.legend(loc=4)
    plt.show()

    fig.savefig('/home/neyo/PycharmProjects/AUC/results/cp_%s.png' % (dataset))