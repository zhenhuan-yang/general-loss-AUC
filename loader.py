import os, psutil
import csv
import numpy as np
import pandas as pd
import h5py

def usage():

    process =psutil.Process(os.getpid())

    mem = process.memory_percent()

    return mem

def loader(filename,n = False,f = False,z = False,m = False,s = True):
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
    with open('/Users/yangzhenhuan/PycharmProjects/AUC/datasets/'+filename, 'r') as file:
        for line in csv.reader(file, delimiter=' '):
            line[0] = '0:' + line[0]
            line = filter(None, line)  # get rid of potential empty elements
            L.append(dict(i.split(':') for i in line))
    print('Done! Memory usage: %f' %(usage()))

    print('Converting to dataframe......', end=' ')
    df = pd.DataFrame(L, dtype='float32').fillna(0) # use float32 to reduce memory usage
    print('Done! Memory usage: %f' % (usage()))

    del L

    print('Converting to array......', end=' ')
    X = df.iloc[:, 1:].values
    Y = df.iloc[:, 0].values.astype('int32')
    print('Done! Memory usage: %f' % (usage()))

    del df

    # centralize
    if m == True:
        print('Nentralizing......', end=' ')
        mean = np.mean(X, axis=1)
        X = (X.transpose() - mean).transpose()
        print('Done! Memory usage: %f' % (usage()))

    # normalize
    if n == True:
        print('Normalizing......', end=' ')
        norm = np.linalg.norm(X, axis=1)
        X = X / norm[:, None]
        print('Done! Memory usage: %f' % (usage()))

    # feature scaling
    if f == True:
        print('Scaling......', end=' ')
        mi = np.min(X,axis=1)
        ma = np.max(X,axis=1)
        X = (X - mi[:, None]) / (ma[:, None] - mi[:, None])
        print('Done! Memory usage: %f' % (usage()))

    # z-sccore
    if z == True:
        print('Z-scoring......', end=' ')
        me = np.mean(X,axis=1)
        st = np.std(X,axis=1)
        X = (X - me[:, None]) / st[:,None]
        print('Done! Memory usage: %f' % (usage()))

    # convert to +1, -1 binary class
    print('Converting to +1 -1 binary class......', end=' ')
    INDEX = np.argwhere(Y == max(Y))
    index = np.argwhere(Y != max(Y))
    Y[INDEX] = 1
    Y[index] = -1
    print('Done! Memory usage: %f' % (usage()))

    # shuffle
    print('Shuffling......', end=' ')
    mask = np.arange(Y.shape[0])
    if s == True:
        np.random.shuffle(mask)
        print('Done! Memory usage: %f' % (usage()))

    return X[mask], Y[mask]

if __name__ == '__main__':
    #np.random.seed(4)
    dataset = 'cod-rna'
    FEATURES,LABELS = loader(dataset)
    print('Write .h5 file......', end=' ')
    hf = h5py.File('/Users/yangzhenhuan/PycharmProjects/AUC/datasets/%s.h5' %(dataset), 'w')
    hf.create_dataset('FEATURES',data=FEATURES)
    hf.create_dataset('LABELS', data=LABELS)
    hf.close()
    print('Done! Memory usage: %f' % (usage()))