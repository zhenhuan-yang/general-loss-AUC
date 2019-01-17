import csv
import numpy as np
import pandas as pd
import h5py

def loader(filename,i,m = True,n = True,s = False):
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
    with open(filename, 'r') as file:
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
        index = np.argwhere(Y == i)
        INDEX = np.argwhere(Y != i)
        Y[index] = -1
        Y[INDEX] = 1
    Y = Y.astype(int)

    # shuffle
    mask = np.arange(Y.shape[0])
    if s == True:
        np.random.shuffle(mask)
    print('Done!')

    return X[mask], Y[mask]

if __name__ == '__main__':
    #np.random.seed(4)
    dataset = 'ijcnn1'
    i = 3
    FEATURES,LABELS = loader(dataset,i,s=True)
    hf = h5py.File('%s.h5' %(dataset), 'w')
    hf.create_dataset('FEATURES',data=FEATURES)
    hf.create_dataset('LABELS', data=LABELS)
    hf.close()