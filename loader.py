import csv
import numpy as np
import pandas as pd
import h5py

def loader(filename,i,m = True,n = True):
    '''
    Data file loader

    input:
        filename - filename

    output:
        x - sample features
        y - sample labels
    '''
    # raw data
    L = []
    with open(filename, 'r') as file:
        for line in csv.reader(file, delimiter=' '):
            line[0] = '0:' + line[0]
            line = filter(None, line)  # get rid of potential empty elements
            L.append(dict(i.split(':') for i in line))
    df = pd.DataFrame(L, dtype=float).fillna(0)
    X = df.iloc[:, 1:].values
    Y = df.iloc[:, 0].values
    # centralize
    if m == True:
        mean = np.mean(X, axis=1)
        X = (X.transpose() - mean).transpose()
    # normalize
    if n == True:
        norm = np.linalg.norm(X, axis=1)
        X = X / norm[:, None]
    # convert to binary class
    if max(Y) == 1:
        print('binary classes already!')
    else:
        r = np.ptp(Y).astype(int)
        print('num of classes: %d' %(r+1))
        index = np.argwhere(Y <= i)
        INDEX = np.argwhere(Y > i)
        Y[index] = -1
        Y[INDEX] = 1
    Y = Y.astype(int)
    return X, Y

if __name__ == '__main__':
    dataset = 'diabetes'
    i = 3
    FEATURES,LABELS = loader(dataset,i,m=False)
    hf = h5py.File('%s.h5' %(dataset), 'w')
    hf.create_dataset('FEATURES',data=FEATURES)
    hf.create_dataset('LABELS', data=LABELS)
    hf.close()