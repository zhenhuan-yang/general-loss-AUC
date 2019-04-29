'''
Bernstein degree
Author: Zhenhuan(Neyo) Yang
'''

import h5py
import matplotlib.pyplot as plt
from SAUC import SAUC

if __name__ == '__main__':

    # Define hyper parameters
    options = {}
    options['name'] = 'hinge'
    options['T'] = 100

    # Define model parameter
    options['N'] = 5
    options['R'] = 1
    options['c'] = 1

    # Define what to run this time
    dataset = 'gisette_scale'

    hf = h5py.File('/home/neyo/PycharmProjects/AUC/h5-datasets/%s.h5' % (dataset), 'r')
    FEATURES = hf['FEATURES'][:]
    LABELS = hf['LABELS'][:]
    hf.close()

    # Simple prepare training and testing
    n = len(LABELS)
    testing = [i for i in range(n // 2)]
    training = [i for i in range(n // 2, n)]

    N = [5,10,15]

    # Prepare results
    res = {}
    for m in N:
        options['N'] = m
        res['%d'%m] = SAUC(FEATURES[training], LABELS[training], FEATURES[testing], LABELS[testing], options)

    # Plot results
    fig = plt.figure()  # create a figure object
    fig.suptitle(dataset)
    for m in N:
        plt.plot(res['%d'%m][0],res['%d'%m][1],label = 'M = %d'%m)
    plt.xlabel('CPU Time(s)')
    plt.ylabel('AUC')
    plt.ylim([.5,1])
    plt.legend(loc=4)
    plt.show()

    fig.savefig('/home/neyo/PycharmProjects/AUC/results/m_%s.png' % (dataset))