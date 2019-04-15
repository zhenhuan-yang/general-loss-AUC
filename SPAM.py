'''
Stochastic Proximal AUC Maximization by natole et al

Author: Zhenhuan(Neyo) Yang
Date: 4/13/19
'''

import numpy as np
import time
from sklearn.metrics import roc_auc_score
from math import sqrt,fabs

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
    lam = options['lam']
    theta = options['theta']
    reg = options['reg']

    print('SPAM with c = %.2f' % (c))

    # get the dimension of what we are working with
    n, d = Xtr.shape

    # initialize
    pt = 0.0
    wt = np.zeros(d)
    at = 0.0
    bt = 0.0
    alphat = 0.0
    Tpt = 0
    Tnt = 0
    mpt = np.zeros(d)
    mnt = np.zeros(d)

    # record auc
    roc_auc = []

    # record time elapsed
    elapsed_time = []
    start_time = time.time()

    for t in range(1,T+1):

        # step size
        eta = c/sqrt(t)

        # approximate prob
        pt = ((t - 1) * pt + (Ytr[t % n] + 1) // 2) / t

        # compute inner product
        prod = np.inner(wt, Xtr[t % n])

        if Ytr[t%n] == 1:
            Tpt += 1
            mpt = ((Tpt - 1) * mpt + Xtr[t%n]) / Tpt
            at = np.inner(wt,mpt)
            alphat = at - bt
            # compute gradient
            gradwt = 2 * (1 - pt) * (prod - at) - 2 * (1 + alphat) * (1 - pt)
        else:
            Tnt += 1
            mnt = ((Tnt - 1) * mnt + Xtr[t%n]) / Tnt
            bt = np.inner(wt,mnt)
            alphat = at - bt
            # compute gradient
            gradwt = 2 * pt * (prod - bt) + 2 * (1 + alphat) * pt

        # update wt
        wt = wt - eta*gradwt*Xtr[t%n]

        # proxima step
        if reg == 'l2':
            wt = prox_l2(wt,lam,eta)
        elif reg == 'net':
            wt = prox_net(wt,lam,theta,eta)
        else:
            print('Wrong regularizer!')
            return

        # write results
        elapsed_time.append(time.time() - start_time)
        roc_auc.append(roc_auc_score(Yte, np.dot(Xte, wt)))

        # running log
        if t % stamp == 0:
            print('iteration: %d AUC: %.6f time eplapsed: %.2f' % (t, roc_auc[-1], elapsed_time[-1]))

    return elapsed_time, roc_auc