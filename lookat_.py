'''
Show results
Author: Zhenhuan(Neyo) Yang
'''

import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from math import fabs

def lookat(algs,names,datasets,para):
    '''
    look at results
    input:
        alg - algorithm
        dataset -
        para - which result you want to see
    output:
        fig -
    '''

    if para == 'cv':

        for dataset in datasets:

            for alg in algs:

                print('alg = %s data = %s' % (alg, dataset))

                # Read
                df = pd.read_pickle('/Users/yangzhenhuan/PycharmProjects/AUC/results/cv_%s_%s.h5' % (alg, dataset))

                for column in df.columns:
                    folders = len(df[column])
                    do = []
                    for folder in range(folders):
                        do.append(max(df[column][folder]))
                    MEAN = np.mean(do)
                    STD = np.std(do)
                    print('c = %.2f R = %.2f AUC = ' % (column[0], column[1]), end=' ')
                    print(('%.4f$\pm$' % MEAN).lstrip('0'), end='')
                    print(('%.4f' % STD).lstrip('0'))


    elif para == 'bern':

        for dataset in datasets:

            # Plot results
            fig = plt.figure()  # create a figure object
            fig.suptitle(dataset)
            plt.style.use('seaborn-whitegrid')

            # Read
            for name in names:
                df = pd.read_pickle('/Users/yangzhenhuan/PycharmProjects/AUC/results/deg_%s_%s.h5' % (name,dataset))

                res = {}
                line = []
                error = []

                for column in df.columns:
                    folders = len(df[column])
                    res[column] = []

                    for folder in range(folders):
                        res[column].append(max(df[column][folder]))

                    line.append(np.mean(res[column]))
                    error.append(np.std(res[column]))

                if name == 'hinge':
                    plt.errorbar(df.columns, line, yerr=error, fmt='r--o', capsize=5, label = name)
                elif name == 'logistic':
                    plt.errorbar(df.columns, line, yerr=error, fmt='b-.*', capsize=5, label = name)

            plt.xlabel('Degree')
            plt.ylabel('AUC')
            # plt.ylim([.5, 1])
            # plt.xticks(df.columns)
            plt.legend(loc=4)
            plt.show()

            fig.savefig('/Users/yangzhenhuan/PycharmProjects/AUC/results/bern_%s.png' % (dataset))

    elif para == 'cp':

        for dataset in datasets:

            # Plot results
            fig = plt.figure()  # create a figure object
            fig.suptitle(dataset)
            for alg in algs:

                # read result
                df = pd.read_pickle('/Users/yangzhenhuan/PycharmProjects/AUC/results/cp_%s_%s.h5' % (alg,dataset))

                if alg == 'SAUC':
                    plt.plot(df['elapsed_time'][:], df['roc_auc'][:], 'b--', label=alg)

                elif alg == 'OAM':
                    plt.plot(df['elapsed_time'][:], df['roc_auc'][:], 'r-', label=alg)

            plt.xlabel('CPU Time(s)')
            plt.ylabel('AUC')
            plt.xlim([-2, 120])
            plt.legend(loc=4)

            plt.show()

            fig.savefig('/Users/yangzhenhuan/PycharmProjects/AUC/results/cp_%s.png' % (dataset))

    else:
        print('Wrong parameter!')
        return

    return

if __name__ == '__main__':

    algs = ['SAUC','OAM']
    names = ['hinge', 'logistic']
    datasets = ['cod-rna','svmguide1','skin_nonskin']

    para = 'bern'

    lookat(algs,names,datasets,para)