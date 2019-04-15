'''
Convergence plot

Author: Zhenhuan(Neyo) Yang
'''

import h5py
import matplotlib.pyplot as plt
from SOLAM import SOLAM
from SAUC import SAUC
from SPAM import SPAM
from OAM import OAM
from OPAUC import OPAUC
from FSAUC import FSAUC

if __name__ == '__main__':

    # Define hyper parameters
    options = {}
    options['T'] = 1000
    options['name'] = 'hinge'
    options['option'] = 'sequential'
    options['sampling'] = 'reservoir'
    options['reg'] = 'l2'

    # Define model parameter
    options['N'] = 10
    options['R'] = 1
    options['c'] = 1
    options['Np'] = 10000
    options['Nn'] = 10000
    options['B'] = 200
    options['delta'] = .05
    options['lam'] = 1
    options['theta'] = 1


    # Define what to run this time
    dataset = 'svmguide3'
    ALG = ['SAUC','OAM']

    hf = h5py.File('/Users/yangzhenhuan/PycharmProjects/AUC/datasets/%s.h5' % (dataset), 'r')
    FEATURES = hf['FEATURES'][:]
    LABELS = hf['LABELS'][:]
    hf.close()

    # Simple prepare training and testing
    n = len(LABELS)
    testing = [i for i in range(n // 2)]
    training = [i for i in range(n // 2, n)]

    # Prepare results
    res = {}
    for alg in ALG:
        if alg == 'SOLAM':
            res[alg] = SOLAM(FEATURES[training], LABELS[training], FEATURES[testing], LABELS[testing], options)
        elif alg == 'FSAUC':
            res[alg] = FSAUC(FEATURES[training], LABELS[training], FEATURES[testing], LABELS[testing], options)
        elif alg == 'OAM':
            res[alg] = OAM(FEATURES[training], LABELS[training], FEATURES[testing], LABELS[testing], options)
        elif alg == 'OPAUC':
            res[alg] = OPAUC(FEATURES[training], LABELS[training], FEATURES[testing], LABELS[testing], options)
        elif alg == 'SPAM':
            res[alg] = SPAM(FEATURES[training], LABELS[training], FEATURES[testing], LABELS[testing], options)
        else:
            res[alg] = SAUC(FEATURES[training], LABELS[training], FEATURES[testing], LABELS[testing], options)

    # Plot results
    fig = plt.figure()  # create a figure object
    fig.suptitle(dataset)
    for alg in ALG:
        plt.plot(res[alg][0],res[alg][1],label = alg)
    plt.xlabel('CPU Time(s)')
    plt.ylabel('AUC')
    # plt.ylim([-1,1])
    plt.legend(loc=4)
    plt.show()

    # fig.savefig('/Users/yangzhenhuan/PycharmProjects/AUC/results/%s.png' % (dataset))