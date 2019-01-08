import numpy as np
from itertools import product
from matplotlib import pyplot as plt

def compute(x):
    '''
    Compute mean and standard deviation
    '''
    BOUND = np.zeros((folders,LAM,C))
    MEAN = np.zeros((LAM, C))
    STD = np.zeros((LAM, C))

    for folder,lam, c in product(range(folders),range(LAM), range(C)):
        BOUND[folder,lam,c] = np.max(x[folder,lam,c,:])
    for lam, c in product(range(LAM), range(C)):
        MEAN[lam, c] = np.mean(BOUND[:,lam,c])
        STD[lam, c] = np.std(BOUND[:,lam,c])

    print('Mean:')
    print(MEAN)
    print('Standard deviation:')
    print(STD)

    return


def draw(ROC_AUC):
    '''
    Plot AUC
    '''
    for folder,lam, c in product(range(folders),range(LAM), range(C)):
        plt.plot(range(T), ROC_AUC[folder, lam, c], label=r'folder = %d $\lambda$ = %d c = %d' % (folder,10**(lam-1),10**(c-1)))
    plt.xlabel('iterations')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()

    return

if __name__ == '__main__':
    dataset = 'diabetes'
    loss = 'hinge'
    x = np.load('%s_%s.npy'%(dataset,loss))
    folders, LAM, C, T = x.shape
    compute(x)
    draw(x)