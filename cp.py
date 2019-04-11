'''
Convergence plot

Author: Zhenhuan(Neyo) Yang
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt
from SAUC import SAUC
from OAM import OAM

if __name__ == '__main__':

    # Define hyper parameters
    options = {}
    options['N'] = 10
    options['name'] = 'hinge'
    options['option'] = 'sequential'
    options['sampling'] = 'sequential'

    # Define model parameter
    options['L'] = 2
    options['c'] = .1
    options['Np'] = 10000
    options['Nn'] = 10000
    options['B'] = 1000

    # Define what to run this time
    dataset = 'a9a'
    ALG = ['SAUC', 'OAM']

    hf = h5py.File('/home/neyo/PycharmProjects/AUC/datasets/%s.h5' % (dataset), 'r')
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
        if alg == 'SAUC':
            options['T'] = 100
            res[alg] = SAUC(FEATURES[training], LABELS[training], FEATURES[testing], LABELS[testing], options)
        else:
            options['T'] = 1000
            res[alg] = OAM(FEATURES[training], LABELS[training], FEATURES[testing], LABELS[testing], options)

    # Plot results
    fig = plt.figure(figsize=(12, 4))  # create a figure object
    fig.suptitle(dataset)
    COLOR = ['r','b']
    for color,alg in zip(COLOR,ALG):
        plt.plot(res[alg][0],res[alg][1],color = color,label = alg)
    plt.xlabel('CPU Time(s)')
    plt.ylabel('AUC')
    plt.legend(loc=4)
    plt.show()
