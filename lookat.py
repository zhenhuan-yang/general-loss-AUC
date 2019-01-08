import numpy as np
from itertools import product
from matplotlib import pyplot as plt

def compute(x):
    folders,GAMMA,LAM,THETA,C = x.shape

    MEAN = np.zeros((GAMMA, LAM, THETA, C))
    STD = np.zeros((GAMMA, LAM, THETA, C))

    for gamma, lam, theta, c in product(range(GAMMA), range(LAM), range(THETA), range(C)):
        MEAN[gamma, lam, theta, c] = np.mean(x[:, gamma, lam, theta, c])
        STD[gamma, lam, theta, c] = np.std(x[:, gamma, lam, theta, c])

    print('Mean:')
    print(MEAN)
    print('Standard deviation:')
    print(STD)

    return


def draw(ROC_AUC):
    '''
    Plot AUC
    '''
    folders, GAMMA, LAM, THETA, C, T = x.shape
    for gamma, lam, theta, c in product(range(GAMMA), range(LAM), range(THETA), range(C)):
        plt.plot(range(T), ROC_AUC[1, gamma, lam, theta, c], label=r'$\gamma$ = %.1f $\lambda$ = %.1f $\theta$ = %.1f c = %.1f' % (gamma,lam,theta,c))
    plt.xlabel('iterations')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()

    return

if __name__ == '__main__':
    dataset = 'splice'
    loss = 'hinge'
    x = np.load('%s_%s.npy'%(dataset,loss))
    draw(x)