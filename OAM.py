'''
Online AUC Maximization by Zhao et al

Author: Zhenhuan(Neyo) Yang

Date: 3/21/19
'''

import numpy as np
import time
from math import log, exp
from sklearn.metrics import roc_auc_score

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

def reservior(Bt,xt,N,M):
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
        Bt.append(xt)
    else:
        z = np.random.binomial(1, p=N/M)
        if z == 1:
            ind = np.random.randint(L)
            Bt[ind] = xt

    return Bt

def seq(loss,wt,xt,yt,B,ct):
    '''
    Sequential update

    input:
        grad - gradient of loss function
        wt - the current classifier
        xt -
        yt -
        B - the buffer to be compared to
        ct - a parameter that weights the comparison

    output:
        wt - th updated classifier
    '''
    L = len(B)
    for i in range(L):
        prod = np.inner(wt,xt - B[i])
        norm = np.inner(xt - B[i],xt - B[i])
        tau = min(ct/2,loss(prod*yt)/norm)
        wt += tau*yt*(xt - B[i])

    return wt

def gra(wt,xt,yt,B,ct):
    '''
    gradient updating

    input:
        wt - the current classifier
        xt -
        yt -
        B - the bufferto be compared to
        ct - a parameter that weights the comparison

    output:
        wt - th updated classifier
    '''
    L = len(B)
    w = wt + 0.0
    for i in range(L):
        prod = np.inner(w,xt - B[i])
        if yt*prod <= 1:
            wt += ct*yt*(xt - B[i])/2

    return wt

def OAM(T,name,option,c,Np,Nn,Xtr,Ytr,Xte,Yte,stamp = 10):
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

    print('OAM......')

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
    # record auc
    roc_auc = []

    # record time elapsed
    elapsed_time = []
    start_time = time.time()

    for t in range(1,T+1):
        if Ytr[t%n] == 1:
            Npt += 1
            ct = c*max(1,Nnt/Nn)
            Bpt = reservior(Bpt,Xtr[t%n],Np,Npt)
            if option == 'sequential':
                wt = seq(loss,wt,Xtr[t%n],Ytr[t%n],Bnt,ct)
            elif option == 'gradient':
                wt = gra(wt, Xtr[t % n], Ytr[t % n], Bnt, ct)
            else:
                print('Wrong update option!')
                return
        else:
            Nnt += 1
            ct = c*max(1,Npt/Np)
            Bnt = reservior(Bnt,Xtr[t%n],Nn,Nnt)
            if option == 'sequential':
                wt = seq(loss,wt,Xtr[t%n],Ytr[t%n],Bpt,ct)
            elif option == 'gradient':
                wt = gra(wt, Xtr[t % n], Ytr[t % n], Bpt, ct)
            else:
                print('Wrong update option!')
                return

        if t % 10 == 0:
            elapsed_time.append(time.time() - start_time)
            roc_auc.append(roc_auc_score(Yte, np.dot(Xte, wt)))
            print('c: %.2f iteration: %d AUC: %.6f time eplapsed: %.2f' % (c, t, roc_auc[-1], elapsed_time[-1]))

            # start_time = time.time()

    return elapsed_time, roc_auc