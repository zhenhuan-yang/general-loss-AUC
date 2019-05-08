'''
Show results
Author: Zhenhuan(Neyo) Yang
'''

import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from math import fabs

def lookat(alg,dataset):
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

    df = pd.read_pickle('/Users/yangzhenhuan/PycharmProjects/AUC/results/cv_%s_%s.h5' % (alg, dataset))

    for column in df.columns:

        print('c = %.2f R = %.2f AUC = ' % (column[0], column[1]), end=' ')
        print(('%.4f$\pm$' % df[column]['MEAN']).lstrip('0'), end='')
        print(('%.4f' % df[column]['STD']).lstrip('0'))

if __name__ == '__main__':

    alg = 'SOLAM'
    dataset = 'a1a'

    lookat(alg,dataset)