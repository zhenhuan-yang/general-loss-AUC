'''
Convert libsvm dataset into normalized binary dataset

Author: Zhenhuan(Neyo) Yang
Date : 5/5/19
'''

from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn import preprocessing
import numpy as np
import os

if __name__ == '__main__':

    datasets = ['webspam_u']

    for dataset in datasets:

        if os.path.isfile('/home/neyo/PycharmProjects/AUC/datasets/%s' % (dataset)):

            print('Loading dataset = %s......' % (dataset), end=' ')
            X, y = load_svmlight_file('/home/neyo/PycharmProjects/AUC/datasets/%s' % (dataset))

            print('Done! Converting to binary......', end=' ')
            m = np.mean(y)
            INDEX = np.argwhere(y > m)
            index = np.argwhere(y <= m)
            y[INDEX] = 1
            y[index] = -1

            print('Done! Normalizing......', end=' ')
            X = preprocessing.normalize(X)

            print('Done! Dumping into file......', end=' ')
            dump_svmlight_file(X, y, '/home/neyo/PycharmProjects/AUC/bi-datasets/%s' % (dataset), zero_based=False)
            print('Done!')

        else:
            pass