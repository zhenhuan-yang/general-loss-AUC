'''
Online AUC Maximization by Zhao et al
Author: Zhenhuan(Neyo) Yang
Date: 3/21/19
'''

import numpy as np
import time
from math import log, exp
from sklearn.metrics import roc_auc_score

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

def loss_func(name):
    '''
    Define loss function
    input:
        name - name of loss function
    output:
        loss - loss function
    '''

    if name == 'hinge':
        loss = lambda x: max(0, 1 - x)
    elif name == 'logistic':
        loss = lambda x: log(1 + exp(-x))
    else:
        print('Wrong loss function!')

    return loss

def reservior(Bt,t,N,M):
    '''
    Reservior Sampling
    input:
        Bt - current buffer
        xt - a training instance
        N - the buffer size
        M - the number of instances received till trial t
    output:
        Bt - updated buffer
    '''

    L = len(Bt)
    if L < N:
        Bt.append(t)
    else:
        z = np.random.binomial(1, p=N/M)
        if z == 1:
            ind = np.random.randint(L)
            Bt[ind] = t

    return Bt

def OAM(Xtr,Ytr,Xte,Yte,options,stamp = 100):
    '''
    Online AUC Maximization
    input:
        T -
        name -
        option - update option
        c - penalty parameter
        Np - maximum buffer size of positive samples
        Nn - maximum buffer size of negative samples
        Xtr -
        Ytr -
        Xte -
        Yte -
        stamp - record stamp
    output:
        elapsed_time -
        roc_auc - auc scores
    '''

    # load parameter
    T = options['T']
    name = options['name']
    option = options['option']
    sampling = options['sampling']
    Np = options['Np']
    Nn = options['Nn']
    c = options['c']
    R = options['R']

    print('OAM with loss = %s sampling = %s option = %s Np = %d Nn = %d R = %.2f c = %.2f' %(name,sampling,option,Np,Nn,R,c))

    # get the dimension of what we are working with
    n, d = Xtr.shape

    # define loss function
    loss = loss_func(name)

    # initialize
    wt = np.zeros(d)
    Bpt = []
    Bnt = []
    Npt = 0
    Nnt = 0

    # restore average wt
    avgwt = wt+0.0

    # record auc
    roc_auc = []

    # record time elapsed
    elapsed_time = []
    start_time = time.time()

    for t in range(1,T+1):
        if Ytr[t%n] == 1:
            Npt += 1
            if sampling == 'reservoir':
                ct = c*max(1,Nnt/Nn)
                Bpt = reservior(Bpt,t%n,Np,Npt)
            elif sampling == 'sequential':
                ct = c
                Bpt.append(t%n)
            else:
                print('wrong sampling option!')
                return
            if option == 'sequential':
                for i in Bnt:
                    prod = wt @ (Xtr[t%n] - Xtr[i])
                    norm = (Xtr[t%n] - Xtr[i]) @ (Xtr[t%n] - Xtr[i])
                    if norm == 0:
                        tau = ct / 2
                    else:
                        tau = min(ct / 2, loss(prod * Ytr[t%n]) / norm)
                    wt += tau * Ytr[t%n] * (Xtr[t%n] - Xtr[i])
                    wt = proj(wt, R)

            elif option == 'gradient':
                w = wt + 0.0
                for i in Bnt:
                    prod = wt @ (Xtr[t%n] - Xtr[i])
                    if Ytr[t%n] * prod <= 1:
                        wt += ct * Ytr[t%n] * (Xtr[t%n] - Xtr[i]) / 2

                wt = proj(wt, R)
            else:
                print('Wrong update option!')
                return
        else:
            Nnt += 1
            if sampling == 'reservoir':
                ct = c*max(1,Npt/Np)
                Bnt = reservior(Bnt,t%n,Nn,Nnt)
            elif sampling == 'sequential':
                ct = c
                Bnt.append(t%n)
            else:
                print('Wrong sampling option!')
                return
            if option == 'sequential':
                for i in Bpt:
                    prod = wt @ (Xtr[t%n] - Xtr[i])
                    norm = (Xtr[t%n] - Xtr[i]) @ (Xtr[t%n] - Xtr[i])
                    if norm == 0:
                        tau = ct / 2
                    else:
                        tau = min(ct / 2, loss(prod * Ytr[t%n]) / norm)
                    wt += tau * Ytr[t%n] * (Xtr[t%n] - Xtr[i])
                    wt = proj(wt, R)
            elif option == 'gradient':
                w = wt + 0.0
                for i in Bpt:
                    prod = wt @ (Xtr[t%n] - Xtr[i])
                    if Ytr[t % n] * prod <= 1:
                        wt += ct * Ytr[t % n] * (Xtr[t % n] - Xtr[i]) / 2
                wt = proj(wt, R)
            else:
                print('Wrong update option!')
                return

        # write results
        elapsed_time.append(time.time() - start_time)
        avgwt = ((t-1)*avgwt + wt) / t
        roc_auc.append(roc_auc_score(Yte, Xte @ avgwt))

        # running log
        if t % stamp == 0:
            print('iteration: %d Buffer: %d AUC: %.6f time eplapsed: %.2f' % (t, len(Bpt)+len(Bnt), roc_auc[-1], elapsed_time[-1]))

    return elapsed_time, roc_auc