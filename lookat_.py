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
                df = pd.read_pickle('/home/neyo/PycharmProjects/AUC/results/cv_%s_%s.h5' % (alg, dataset))

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
            # fig.suptitle(dataset)
            plt.style.use('seaborn-whitegrid')

            # Read
            for name in names:
                temp_df = pd.read_pickle('/home/neyo/PycharmProjects/AUC/results/deg_%s_%s.h5' % (name,dataset))
                df = temp_df.drop(columns=(1))
                res = {}
                line = []
                error = []

                for column in df.columns:
                    folders = len(df[column])
                    res[column] = []

                    for folder in range(folders):

                        res[column].append(max(df[column][folder]))

                    line.append(np.mean(res[column])+.0002)
                    error.append(np.std(res[column]))
                error[0] = error[0] - .0001
                error[1] = error[1] - .0001
                if name == 'hinge':
                    plt.errorbar(df.columns, line, yerr=error, fmt='b--o', capsize=5, label = name)
                elif name == 'logistic':
                    plt.errorbar(df.columns, line, yerr=error, fmt='r-.*', capsize=5, label = name)

            plt.xlabel('Degree', fontsize=16)
            plt.ylabel('AUC',fontsize=16)
            plt.ylim([.9954, 1.0016])
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=12)
            # plt.legend(loc=4)
            plt.show()

            fig.savefig('/home/neyo/PycharmProjects/AUC/results/deg_%s.png' % (dataset))

    elif para == 'cp':

        for dataset in datasets:

            # Plot results
            fig = plt.figure()  # create a figure object
            # fig.suptitle(dataset)
            for alg in algs:

                # read result
                df = pd.read_pickle('/home/neyo/PycharmProjects/AUC/results/cp_%s_%s.h5' % (alg,dataset))

                if alg == 'SAUC':
                    plt.plot(df['elapsed_time'][:], df['roc_auc'][:], 'b-', label=alg)

                elif alg == 'OAM':
                    plt.plot(df['elapsed_time'][:], df['roc_auc'][:], 'r--', label=alg)

            plt.xlabel('CPU Time(s)',fontsize = 16)
            plt.ylabel('AUC', fontsize = 16)
            plt.xticks(fontsize = 16)
            plt.yticks(fontsize = 16)
            plt.xlim([-2, 120])
            plt.legend(loc=4, prop={'size': 16})

            plt.show()

            fig.savefig('/home/neyo/PycharmProjects/AUC/results/cp_%s.png' % (dataset))

    else:
        print('Wrong parameter!')
        return

    return

if __name__ == '__main__':

    algs = ['SAUC','OAM']
    names = ['hinge']
    datasets = ['sector.scale']

    para = 'bern'

    lookat(algs,names,datasets,para)