'''
Stochastic Proximal AUC Maximization by Natole et al
Author: Zhenhuan(Neyo) Yang
Date: 4/13/19
'''

import numpy as np
import time
from sklearn.metrics import roc_auc_score
from math import sqrt,fabs

def proj(x, R):
    '''
    Projection
    input:
        x -
        R - radius
    output:
        proj - projected
    '''
    norm = np.linalg.norm(x)
    if norm > R:
        x = x / norm * R
    return x

def prox_l2(x,lam,eta):
    '''
    L2 proximal
    input:
        x -
        lam - parameter
        eta - step size
    output:
        x -
    '''
    x = x / (1+lam*eta)
    return x

def prox_net(x,lam,theta,eta):
    '''
    Elastic net proximal
    input:
        x -
        lam - l2 parameter
        theta - l1 parameter
        eta -  step size
    output:
        x -
    '''
    x = np.sign(x) * max(0,fabs(x)/(eta*lam+1)) - eta*theta/(eta*lam+1)

    return x

def SPAM(Xtr,Ytr,Xte,Yte,options,stamp=100):
    '''
    Stochastic Proximal AUC Maximization
    input:
        Xtr -
        Ytr -
        Xte -
        Yte -
        options -
    output:
        elapsed_time -
        roc_auc -
    '''

    # load parameter
    T = options['T']
    c = options['c']
    R = options['R'] # modified algorithm bounded not regularized
    theta = options['theta']
    reg = options['reg']

    print('SPAM with R = %.2f c = %.2f' % (R,c))

    # get the dimension of what we are working with
    n, d = Xtr.shape

    # initialize
    pt = sum(Ytr[Ytr == 1]) / n
    wt = np.zeros(d)
    mpt = np.mean(Xtr[Ytr == 1],axis=0)
    mnt = np.mean(Xtr[Ytr == -1],axis=0)

    # record auc
    roc_auc = []

    # record time elapsed
    elapsed_time = []
    start_time = time.time()

    for t in range(1,T+1):

        # step size
        eta = c/sqrt(t)

        # compute inner product
        prod = np.inner(wt, Xtr[t % n])

        # compute a,b,alpha
        at = np.inner(wt, mpt)
        bt = np.inner(wt, mnt)
        alphat = at - bt

        # compute gradient
        if Ytr[t%n] == 1:
            gradwt = 2 * (1 - pt) * (prod - at) - 2 * (1 + alphat) * (1 - pt)
        else:
            gradwt = 2 * pt * (prod - bt) + 2 * (1 + alphat) * pt

        # update wt
        wt = proj(wt - eta*gradwt*Xtr[t%n], R)

        # proxima step
        # if reg == 'l2':
        #     wt = prox_l2(wt,lam,eta)
        # elif reg == 'net':
        #     wt = prox_net(wt,lam,theta,eta)
        # else:
        #     print('Wrong regularizer!')
        #     return

        # write results
        elapsed_time.append(time.time() - start_time)
        roc_auc.append(roc_auc_score(Yte, np.dot(Xte, wt)))

        # running log
        if t % stamp == 0:
            print('iteration: %d AUC: %.6f time eplapsed: %.2f' % (t, roc_auc[-1], elapsed_time[-1]))

    return elapsed_time, roc_auc