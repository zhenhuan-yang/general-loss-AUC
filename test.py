'''
Convergence plot
Author: Zhenhuan(Neyo) Yang
'''

import h5py
import matplotlib.pyplot as plt
import SAUC
import SAUC_test
import SAUC_prev
import SAUC_new

if __name__ == '__main__':

    # Define hyper parameters
    options = {}
    options['T'] = 100
    options['name'] = 'hinge'
    options['option'] = 'sequential'
    options['sampling'] = 'reservoir'
    options['reg'] = 'l2'

    # Define model parameter
    options['N'] = 5
    options['R'] = 1
    options['c'] = 1
    options['Np'] = 10000
    options['Nn'] = 10000
    options['B'] = 200
    options['delta'] = .05
    options['lam'] = 1
    options['theta'] = 1


    # Define what to run this time
    dataset = 'a1a'
    ALG = ['prev','test','now']

    hf = h5py.File('/home/neyo/PycharmProjects/AUC/h5-datasets/%s.h5' % (dataset), 'r')
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
        if alg == 'now':
            res[alg] = SAUC.SAUC(FEATURES[training], LABELS[training], FEATURES[testing], LABELS[testing], options)
        elif alg == 'prev':
            res[alg] = SAUC_prev.SAUC(FEATURES[training], LABELS[training], FEATURES[testing], LABELS[testing], options)
        elif alg == 'new':
            res[alg] = SAUC_new.SAUC(FEATURES[training], LABELS[training], FEATURES[testing], LABELS[testing], options)
        elif alg == 'test':
            res[alg] = SAUC_test.SAUC(FEATURES[training], LABELS[training], FEATURES[testing], LABELS[testing], options)
        else:
            pass


    # Plot results
    fig = plt.figure()  # create a figure object
    fig.suptitle(dataset)
    for alg in ALG:
        plt.plot(res[alg][0],res[alg][1],label = alg)
    plt.xlabel('iteration')
    plt.ylabel('AUC')
    # plt.ylim([-1,1])
    plt.legend(loc=4)
    plt.show()

    # fig.savefig('/home/neyo/PycharmProjects/AUC/results/%s.png' % (dataset))