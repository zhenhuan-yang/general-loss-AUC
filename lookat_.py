'''
Show results
Author: Zhenhuan(Neyo) Yang
'''

import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from math import fabs

def lookat(algs,datasets,para):
    '''
    look at results

    input:
        alg - algorithm
        dataset -
        para - which result you want to see

    output:
        fig -
    '''

    # result = pd.DataFrame()
    for dataset in datasets:

        for alg in algs:

            print('alg = %s data = %s' % (alg, dataset))

            if para == 'bound':

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

                # Read
                df = pd.read_pickle('/home/neyo/PycharmProjects/AUC/results/deg_%s.h5' % (dataset))

                # results
                line = []
                error = []
                for m in df.columns:
                    ind = np.argmax(df[m]['MEAN'])
                    line.append(df[m]['MEAN'][ind])
                    error.append(df[m]['STD'][ind])
                plt.style.use('seaborn-whitegrid')
                plt.errorbar(df.columns, line, yerr=error, fmt='--o', capsize=5)

                plt.xlabel('Degree')
                plt.ylabel('AUC')
                plt.ylim([.5, 1])
                plt.xticks(df.columns)
                plt.title('%s' % (dataset))
                plt.show()

            else:
                print('Wrong parameter!')
                return

    return

if __name__ == '__main__':

    algs = ['SPAM']
    datasets = ['a1a']
    para = 'bound'

    lookat(algs,datasets,para)
