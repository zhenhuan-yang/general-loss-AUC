'''
Implement following algorithms:
    SAUC
    OAM

Author: Zhenhuan(Neyo) Yang
'''

import h5py
from matplotlib import pyplot as plt
from math import fabs
import SAUC
import OAM

def split(folder,folders):

    '''
    Split the dataset by indices

    input:
        folder - current folder
        folders - number of folders

    output:
        train_list -
        test_list -
    '''

    if folder >= folders:
        print('Exceed maximum folders!')
        return

    # regular portion of each folder
    portion = round(n / folders)
    start = portion * folder
    stop = portion * (folder + 1)

    if folders == 1:
        train_list = [i for i in range(n)]
        test_list = [i for i in range(n)]

    elif folders == 2:
        if folder == 0:
            train_list = [i for i in range(start)] + [i for i in range(stop, n)]
            test_list = [i for i in range(start, stop)]
        else:
            train_list = [i for i in range(start)]
            test_list = [i for i in range(start, n)]

    else:
        if fabs(stop - n) < portion:  # remainder occurs
            train_list = [i for i in range(start)]
            test_list = [i for i in range(start, n)]
        else:
            train_list = [i for i in range(start)] + [i for i in range(stop, n)]
            test_list = [i for i in range(start, stop)]

    return train_list, test_list

if __name__ == '__main__':

    # Read data from hdf5 file
    dataset = 'covtype'
    hf = h5py.File('/Users/yangzhenhuan/PycharmProjects/AUC/datasets/%s.h5' % (dataset), 'r')
    FEATURES = hf['FEATURES'][:]
    LABELS = hf['LABELS'][:]
    hf.close()

    # Define hyper parameters
    N = 3
    T = 500
    folders = 2

    # Define model parameters
    L = [1]
    C = [1]
    Np = [100]
    Nn = [100]

    # Define losses and algorithms
    NAME = ['hinge']
    ALG = ['SAUC','OAM']
    OPTION = ['sequential','gradient']

    # Prepare training and testing
    n = len(LABELS)
    training = [i for i in range(n // 2)]
    testing = [i for i in range(n // 2, n)]

    # Prepare results
    sauc_time,sauc_auc = SAUC.SAUC(T,NAME[0],N,L[0],C[0],FEATURES[training],LABELS[training],FEATURES[testing],LABELS[testing])
    oam_time,oam_auc = OAM.OAM(T,NAME[0],OPTION[0],C[0],Np[0],Nn[0],FEATURES[training],LABELS[training],FEATURES[testing],LABELS[testing])

    fig = plt.figure()  # create a figure object
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(sauc_time, sauc_auc, label='SAUC')
    ax.plot(oam_time, oam_auc, label='OAM')
    ax.set_xlabel('Time')
    ax.set_ylabel('AUC')
    plt.legend()
    plt.show()