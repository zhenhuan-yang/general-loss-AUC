'''
Load dataset from raw into .h5 file
'''

import csv
import numpy as np
import pandas as pd
import h5py

def loader(filename,i,n = True,f = False,z = False,m = True,s = True):
    '''
    Data file loader

    input:
        filename - filename

    output:
        x - sample features
        y - sample labels
    '''
    print('Initializing cvs reader......', end=' ')
    # raw data
    L = []
    with open('/home/neyo/PycharmProjects/AUC/datasets/%s' % filename, 'r') as file:
        for line in csv.reader(file, delimiter=' '):
            line[0] = '0:' + line[0]
            line = filter(None, line)  # get rid of potential empty elements
            L.append(dict(i.split(':') for i in line))
    print('Done!')
    print('Converting to dataframe......', end=' ')
    df = pd.DataFrame(L, dtype=float).fillna(0)
    print('Done!')
    print('Converting to array......', end=' ')
    X = df.iloc[:, 1:].values
    Y = df.iloc[:, 0].values
    num,d = X.shape
    print('number of samples: %d number of features: %d' %(num,d))

    # centralize
    if m == True:
        mean = np.mean(X, axis=1)
        X = (X.transpose() - mean).transpose()

    # normalize
    if n == True:
        norm = np.linalg.norm(X, axis=1)
        X = X / norm[:, None]

    # feature scaling
    if f == True:
        mi = np.min(X,axis=1)
        ma = np.max(X,axis=1)
        X = (X - mi[:, None]) / (ma[:, None] - mi[:, None])

    # z-sccore
    if z == True:
        me = np.mean(X,axis=1)
        st = np.std(X,axis=1)
        X = (X - me[:, None]) / st[:,None]

    # convert to binary class
    r = np.ptp(Y).astype(int)
    if r == 1:
        print('binary classes already!')
        index = np.argwhere(Y == 1)
        INDEX = np.argwhere(Y != 1)
        Y[index] = 1
        Y[INDEX] = -1

    else:
        print('num of classes: %d' %(r+1))
        index = np.argwhere(Y == i)
        INDEX = np.argwhere(Y != i)
        Y[index] = 1
        Y[INDEX] = -1
    Y = Y.astype(int)

    # shuffle
    mask = np.arange(num)
    if s == True:
        np.random.shuffle(mask)
    print('Done!')

    return X[mask], Y[mask]

if __name__ == '__main__':
    #np.random.seed(4)
    dataset = 'news20.binary'
    i = 3
    FEATURES,LABELS = loader(dataset,i)
    hf = h5py.File('/home/neyo/PycharmProjects/AUC/datasets/%s.h5' %(dataset), 'w')
    hf.create_dataset('FEATURES',data=FEATURES)
    hf.create_dataset('LABELS', data=LABELS)
    hf.close()