'''
Convert libsvm dataset into normalized binary dataset

Author: Zhenhuan(Neyo) Yang
Date : 5/5/19
'''

from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn import preprocessing
import numpy as np

if __name__ == '__main__':

    dataset = 'dna'
    X, y = load_svmlight_file('/home/neyo/PycharmProjects/AUC/datasets/%s' % (dataset))

    INDEX = np.argwhere(y == max(y))
    index = np.argwhere(y != max(y))
    y[INDEX] = 1
    y[index] = -1

    X = preprocessing.normalize(X)

    dump_svmlight_file(X, y, '/home/neyo/PycharmProjects/AUC/bi-datasets/%s' % (dataset), zero_based=False)