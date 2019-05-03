'''
Show results
Author: Zhenhuan(Neyo) Yang
'''

import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from math import fabs

def lookat(alg,dataset,para):
    '''
    look at results

    input:
        alg - algorithm
        dataset -
        para - which result you want to see

    output:
        fig -
    '''

    print('alg = %s data = %s' %(alg,dataset))

    # result = pd.DataFrame()

    if para == 'bound':
        # Read
        df = pd.read_pickle('/home/neyo/PycharmProjects/AUC/results/cv_%s_%s.h5' % (alg, dataset))

        for column in df.columns:
            ind = np.argmax(df[column]['MEAN'])
            last1 = df[column]['MEAN'][-1]
            last2 = df[column]['MEAN'][-9]

            if fabs(last1 - last2) > .001 or ind < 25:
                pass
            else:
                print('c = %.2f R = %.2f AUC = ' % (column[0], column[1]), end=' ')
                print(('%.4f$\pm$' % df[column]['MEAN'][ind]).lstrip('0'), end='')
                print(('%.4f' % df[column]['STD'][ind]).lstrip('0'))

                plt.plot(df[column]['MEAN'], label='c= %.2f R = %.2f AUC = %.4f$\pm$%.4f$'
                                                   % (column[0], column[1], df[column]['MEAN'][ind], df[column]['STD'][ind]))
        # Results
        # for column in df.columns:
        #     result[column] = [np.max(df[column]['MEAN'])]
        #     c = column[0]
        #     r = column[1]
        #     plt.plot(df[column]['MEAN'], label='c= %.2f R = %.2f AUC = %.4f$\pm$' % (c, r, result[column]))
        #
        # col_ind = np.argmax(result.values)
        # col = result.columns[col_ind]
        # ind = np.argmax(df[col]['MEAN'])
        # print('alg = %s data = %s c = %.2f R = %.2f AUC = ' % (alg, dataset, col[0], col[1]), end=' ')
        # print(('%.4f$\pm$' % result[col]).lstrip('0'), end='')
        # print(('%.4f' % df[col]['STD'][ind]).lstrip('0'))
        plt.xlabel('Iteration')
        plt.ylabel('AUC')
        plt.ylim([.5, 1])
        plt.legend(loc=4,prop={'size': 6})
        plt.title('%s_%s' % (alg, dataset))
        plt.show()

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
        plt.errorbar(df.columns,line,yerr=error,fmt='--o',capsize=5)

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

    alg = 'OAM'
    dataset = 'leu'
    para = 'bound'

    lookat(alg,dataset,para)