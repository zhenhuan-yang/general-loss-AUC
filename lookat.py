import numpy as np
from itertools import product
from matplotlib import pyplot as plt

def compute(x):
    '''
    Compute mean and standard deviation
    '''
    BOUND = np.zeros((folders,LOSS,ALG,L,LAM,C))
    MEAN = np.zeros((LOSS,ALG,L,LAM, C))
    STD = np.zeros((LOSS,ALG,L,LAM, C))

    for folder,loss,alg,l,lam, c in product(range(folders),range(LOSS),range(ALG),range(L),range(LAM), range(C)):
        BOUND[folder,loss,alg,l,lam,c] = np.max(x[folder,loss,alg,l,lam,c,:])
    for loss,alg,l,lam, c in product(range(LOSS),range(ALG),range(L),range(LAM), range(C)):
        MEAN[loss,alg,l,lam, c] = np.mean(BOUND[:,loss,alg,l,lam,c])
        STD[loss,alg,l,lam, c] = np.std(BOUND[:,loss,alg,l,lam,c])

    print('Mean:')
    print(MEAN)
    print('Standard deviation:')
    print(STD)

    return


def draw(ROC_AUC):
    '''
    Plot AUC
    '''
    for folder,loss,alg,l,lam, c in product(range(folders),range(LOSS),range(ALG),range(L),range(LAM), range(C)):
        plt.plot(range(T), ROC_AUC[folder, loss, alg, l, lam, c], label=r'folder = %d loss = %d alg = %d l = %d $\lambda$ = %d c = %d' % (folder,loss,alg,l,lam,c))
    plt.xlabel('iterations')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()

    return

if __name__ == '__main__':
    dataset = 'covtype'
    x = np.load('%s.npy'%(dataset))
    folders, LOSS, ALG, L, LAM, C, T = x.shape
    compute(x)
    draw(x)