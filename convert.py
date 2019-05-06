'''
Convert libsvm dataset into normalized binary dataset

Author: Zhenhuan(Neyo) Yang
Date : 5/5/19
'''

from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn import preprocessing
import numpy as np

if __name__ == '__main__':

    datasets = ['a1a','a2a','a3a','a4a','a5a','a6a','a7a','a8a','a9a','acoustic','australian_scale','avazu-app',
                'breast-cancer_scale','cifar10','cod-rna','combined_scale','connect-4','covtype.libsvm.binary.scale',
                'diabetes_scale','dna','epsilon_normalized','fourclass_scale','german.numer_scale','gisette_scale','glass',
                'heart_scale','HIGGS','ijcnn1','ionosphere_scale','iris','letter','leu','liver-disorders_scale',
                'madelon','mnist.scale','mnist8m.scale','mushroom','news20.binary','pendigits','poker','protein',
                'rcv1_train.binary','real-sim','satimage','sector','sector.scale','segment','seismic','sensorless',
                'shuttle','skin_nonskin','smallNORB','sonar_scale','splice_scale','SUSY','SVHN.scale',
                'svmguide1','svmguide2','svmguide3','svmguide4','url_combined','usps','vehicle','vowel',
                'w1a','w2a','w3a','w4a','w5a','w6a','w7a','w8a','wine.scale']
    
    for dataset in datasets:
        
        print('Loading dataset = %s......' %(dataset), end = ' ')
        X, y = load_svmlight_file('/home/neyo/PycharmProjects/AUC/datasets/%s' % (dataset))
        
        print('Done! Converting to binary......', end = ' ')
        INDEX = np.argwhere(y == max(y))
        index = np.argwhere(y != max(y))
        y[INDEX] = 1
        y[index] = -1
        
        print('Done! Normalizing......', end = ' ')
        X = preprocessing.normalize(X)
        
        print('Done! Dumping into file......', end = ' ')
        dump_svmlight_file(X, y, '/home/neyo/PycharmProjects/AUC/bi-datasets/%s' % (dataset), zero_based=False)
        print('Done!')
