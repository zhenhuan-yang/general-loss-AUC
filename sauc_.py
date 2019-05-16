"""
Created on Tue Nov  6 09:25:13 2018
@author: Zhenhuan Yang
# -*- coding: utf-8 -*-
Spyder Editor
We apply the algorithm in Yang, 2019 NIPS to do Fast AUC maximization
Input:
    x_tr: training instances
    y_tr: training labels
    x_te: testing instances
    y_te: testing labels
    options: a dictionary
        'ids' stores the indices of examples traversed, ids divided by number of training examples is the number of passes
        'c' stores the initial step size
        'beta': the parameter R
        'n_pass': the number of passes
        'time_lim': optional argument, the maximal time allowed
Output:
    roc_auc: results on iterates indexed by res_idx
    time:
"""

import numpy as np
from math import factorial
from sklearn.metrics import roc_auc_score
import time

def loss_func(name, L):

    '''
    Define loss function
    input:
        name - name of loss funtion
        R - bound for w
    output:
        loss - loss function
    '''

    if name == 'hinge':
        loss = lambda x: max(0, 1 + L - 2 * L * x)
    elif name == 'logistic':
        loss = lambda x: np.log(1 + np.exp(L - 2 * L * x))
    else:
        print('Wrong loss function!')

    return loss

def comb(N):
    '''
    Compute combination
    input:
        N - degree of Bernstein
    output:
        c - combination dictionary
    '''

    c = {}
    for n in range(N + 1):
        c[n] = np.zeros(n + 1)
        for k in range(n + 1):
            c[n][k] = factorial(n) / factorial(k) / factorial(n - k)
    return c

def bound(N, loss, L, comb_dict):

    '''
    Calculate annoying parameters to estimate gamma
    '''

    R1 = 0.0
    R2 = 0.0
    Sp1 = 0.0
    Sm1 = 0.0
    Sp2 = 0.0
    Sm2 = 0.0
    for i in range(N + 1):
        # compute plus
        alpha0 = L ** i
        alpha1 = i * L ** (i - 1)
        alpha2 = i * (i - 1) * L ** (i - 2)
        R1 += alpha0
        Sp1 += alpha1
        Sp2 += alpha2
        # compute minus
        beta0 = 0.0
        beta1 = 0.0
        beta2 = 0.0
        for k in range(i, N + 1):
            # compute forward difference
            delta = 0.0
            for j in range(k + 1):
                delta += comb_dict[k][j] * (-1) ** (k - j) * loss(j / N)
            # compute coefficient
            beta0 += comb_dict[N][k] * comb_dict[k][i] * (N + 1) * abs(delta) / (2 ** k) / (L ** i)
            beta1 += comb_dict[N][k] * comb_dict[k][i] * (N + 1) * (k - i) * abs(delta) / (2 ** k) / (L ** (i + 1))
            beta2 += comb_dict[N][k] * comb_dict[k][i] * (N + 1) * (k - i) * (k - i - 1) * abs(delta) / (2 ** k) / (
                        L ** (i + 2))
        R2 += beta0
        Sm1 += beta1
        Sm2 += beta2

    gamma = max((2 * R1 + R2) * Sp2 + Sp1 ** 2, (2 * R2 + R1) * Sm2 + Sm1 ** 2) / (N + 1)

    return R1, R2, gamma

def coef(N, loss, L, comb_dict):
    '''
    Compute the coefficient first
    input:
        N - degree of Bernstein
        loss - loss function
        L -
        comb_dict -
    output:
        beta - coefficient dictionary
        gbeta - gradient coefficient dictionary
    '''

    beta = {}
    gbeta = {}
    for i in range(N + 1):
        beta[i] = np.zeros(N - i + 1)
        gbeta[i] = np.zeros(N - i + 1)
        for k in range(i, N + 1):
            # compute forward difference
            delta = 0.0
            for j in range(k + 1):
                delta += comb_dict[k][j] * (-1) ** (k - j) * loss(j / N)
            # compute coefficient
            beta[i][k - i] = comb_dict[N][k] * comb_dict[k][i] * (N + 1) * delta / ((2 * L) ** k)
            gbeta[i][k - i] = beta[i][k - i] * (k - i)

    return beta, gbeta

def pos(N, prod, L):
    '''
    Compute positive function and gradient information
    input:
        i - index of function
        prod - wt*xt
        L - bound on prod
    output:
        fpt - positive function value
        gfpt - positive function gradient
    '''

    plus = L / 2 + prod
    p = list(range(N + 1))
    fpt = np.power(plus, p)
    gfpt = np.multiply(fpt, p) / plus  # no xt yet!

    return fpt, gfpt

def neg(N, prod, L, beta, gbeta):
    '''
    Compute negative function and gradient information
    input:
        N -
        loss - loss function
        prod - wt*xt
        L -
        beta - coefficient
    output:
        fnt - negative function value
        gfnt - negative function gradient
        wasted - time wasted on computing
    '''

    minus = L / 2 - prod
    p = list(range(N + 1))
    # exponent
    exponent = np.power(minus, p)

    fnt = np.zeros(N + 1)
    gfnt = np.zeros(N + 1)

    for i in range(N + 1):
        # compute function value
        fnt[i] = np.inner(beta[i], exponent[:N - i + 1])

        # compute gradient
        gfnt[i] = np.inner(gbeta[i], exponent[:N - i + 1]) / minus  # no xt yet!

    return fnt, gfnt

def SAUC(Xtr,Xte,Ytr,Yte,options,stamp = 10):
    '''
    Stochastic AUC Optimization with General Loss
    input:
        T -
        name -
        N - Bernstein degree
        L - Bound for prod
        c - step size parameter
        Xtr - Training features
        Ytr - Training labels
        Xte - Testing features
        Yte - Testing labels
        stamp - record stamp
    output:
        elapsed_time -
        roc_auc - auc scores
    '''

    # load parameter
    ids = options['ids']
    n_ids = len(ids)
    T = int(round((-1 + np.sqrt(1 + 8 * n_ids)) / 2) - 1)  # outer loop iterations
    name = options['name']
    N = options['m']
    R = options['R']
    L = 2 * R
    c = options['c']

    # get the dimension of what we are working with
    n, d = Xtr.shape

    WT = np.zeros(d)
    AT = np.zeros(N + 1)
    BT = np.zeros(N + 1)
    ALPHAT = np.zeros(N + 1)

    # define loss function
    loss = loss_func(name, L)

    # compute combinations and coefficients
    comb_dict = comb(N)
    beta, gbeta = coef(N, loss, L, comb_dict)

    # compute gamma
    R1, R2, gamma = bound(N,loss,L,comb_dict)

    print('SAUC with loss = %s N = %d R = %d gamma = %.02f c = %d' % (name, N, R, gamma, c))

    # restore average wt
    # avgwt = WT + 0.0

    # record auc
    roc_auc = []

    k = 0

    # record time elapsed
    elapsed_time = []
    elapsed = 0
    start_time = time.time()

    # Begin algorithm
    for t in range(1, T + 1):
        # initialize inner loop variables
        wj = WT + 0.0
        aj = AT + 0.0
        bj = BT + 0.0
        alphaj = ALPHAT + 0.0

        BWt = 0.0
        BAt = 0.0
        BBt = 0.0
        BALPHAt = 0.0

        # step size
        eta = c / np.sqrt(t) / gamma

        # inner loop update at j
        for j in range(t):

            xt = Xtr[ids[k]]
            yt = Ytr[ids[k]]

            k += 1

            prod = xt @ wj

            fpt, gfpt = pos(N, prod, L)
            fnt, gfnt = neg(N, prod, L, beta, gbeta)

            # if condition is faster than two inner product!
            if yt == 1:
                gradwt = 2 * (alphaj - aj) @ gfpt
                gradat = 2 * (aj - fpt)
                gradbt = 2 * bj
                gradalphat = -2 * (alphaj - fpt)
            else:
                gradwt = 2 * (alphaj - bj) @ gfnt
                gradat = 2 * aj
                gradbt = 2 * (bj - fnt)
                gradalphat = -2 * (alphaj - fnt)

            wj = wj - eta * (gradwt * xt * yt / (2 * (N + 1)) + gamma * (wj - WT))
            aj = aj - eta * gradat / (2 * (N + 1))
            bj = bj - eta * gradbt / (2 * (N + 1))
            alphaj = alphaj + eta * gradalphat / (2 * (N + 1))

            # some projection
            # ---------------------------------
            tnm = np.linalg.norm(wj)
            if tnm > R:
                wj = wj * (R / tnm)
            tnm = np.linalg.norm(aj)
            if tnm > R1:
                aj = aj * (R1 / tnm)
            tnm = np.linalg.norm(bj)
            if tnm > R2:
                bj = bj * (R2 / tnm)
            tnm = np.linalg.norm(alphaj)
            if tnm > R1 + R2:
                alphaj = alphaj * ((R1 + R2) / tnm)
            # ---------------------------------

            BWt += wj
            BAt += aj
            BBt += bj
            BALPHAt += alphaj

        # update outer loop variables
        WT = BWt / t
        AT = BAt / t
        BT = BBt / t
        ALPHAT = BALPHAt / t

        stop_time = time.time()

        elapsed += stop_time - start_time
        elapsed_time.append(elapsed)
        # avgwt = ((t - 1) * avgwt + WT) / t

        roc_auc.append(roc_auc_score(Yte, Xte @ WT))

        if t % stamp == 0 or t == T:
            print('iteration: %d AUC: %.6f time elapsed: %.2f' % (t, roc_auc[-1], elapsed_time[-1]))

        start_time = time.time()

    return elapsed_time, roc_auc