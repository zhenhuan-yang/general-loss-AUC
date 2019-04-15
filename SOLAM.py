'''
Stochastic Online AUC Maximization

Author: Zhenhuan(Neyo) Yang
'''

import numpy as np
import time
from sklearn.metrics import roc_auc_score
from math import sqrt

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

def SOLAM(Xtr, Ytr, Xte, Yte, options,stamp = 100):
    '''
    Stochastic Online AUC Maximization

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
    R = options['R']
    L = 2 * R * max(np.linalg.norm(Xtr, axis=1))

    print('SOLAM with c = %.2f' % (c))

    # get the dimension of what we are working with
    n, d = Xtr.shape

    # initialize
    pt = 0.0
    wt = np.zeros(d)
    at = 0.0
    bt = 0.0
    alphat = 0.0
    bwt = np.zeros(d)
    bat = 0.0
    bbt = 0.0
    balphat = 0.0
    beta = 0.0

    # record auc
    roc_auc = []

    # record time elapsed
    elapsed_time = []
    start_time = time.time()

    for t in range(1,T+1):

        # approximate prob
        pt = ((t-1)*pt + (Ytr[t%n]+1)//2)/t

        # compute inner product
        prod = np.inner(wt,Xtr[t%n])

        # step size
        eta = c/sqrt(t)

        # compute gradient
        if Ytr[t%n] == 1:
            gradwt = 2*(1-pt)*(prod - at) - 2*(1+alphat)*(1-pt)
            gradat = 2*(1-pt)*(at - prod)
            gradbt = 0.0
            gradalphat = -2*(1-pt)*prod - 2*pt*(1-pt)*alphat
        else:
            gradwt = 2 * pt * (prod - bt) + 2 * (1 + alphat) * pt
            gradat = 0.0
            gradbt = 2*pt*(bt-prod)
            gradalphat = 2 * pt * prod - 2 * pt * (1 - pt) * alphat
        # update variable
        wt = proj(wt - eta*gradwt*Xtr[t%n],R)
        at = proj(at - eta*gradat,L/2)
        bt = proj(bt - eta*gradbt,L/2)
        alphat = proj(alphat + eta*gradalphat,L)

        # update output
        bwt = (beta*bwt + eta*wt)/(beta+eta)
        bat = (beta * bat + eta * at) / (beta + eta)
        bbt = (beta * bbt + eta * bt) / (beta + eta)
        balphat = (beta * balphat + eta * alphat) / (beta + eta)
        beta += eta

        # write results
        elapsed_time.append(time.time() - start_time)
        roc_auc.append(roc_auc_score(Yte, np.dot(Xte, bwt)))

        # running log
        if t % stamp == 0:
            print('iteration: %d AUC: %.6f time eplapsed: %.2f' % (t, roc_auc[-1], elapsed_time[-1]))

    return elapsed_time, roc_auc