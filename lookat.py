import numpy as np
from itertools import product

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

if __name__ == '__main__':
    usps_logisitc = np.load('usps_logisitc.npy')
    compute(usps_logisitc)