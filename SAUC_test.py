'''
Stochastic AUC Optimization with General Loss by Yang et al
Author: Zhenhuan(Neyo) Yang
Date: 4/8/19
'''

import numpy as np
from math import fabs, sqrt, log, exp, factorial
import time
from sklearn.metrics import roc_auc_score

def bound(N, loss, L, comb_dict):
    '''
    Calculate annoying parameters to estimate rho
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
            beta0 += comb_dict[N][k] * comb_dict[k][i] * (N + 1) * fabs(delta) / (2 ** k) / (L ** i)
            beta1 += comb_dict[N][k] * comb_dict[k][i] * (N + 1) * (k - i) * fabs(delta) / (2 ** k) / (L ** (i + 1))
            beta2 += comb_dict[N][k] * comb_dict[k][i] * (N + 1) * (k - i) * (k - i - 1) * fabs(delta) / (2 ** k) / (
                        L ** (i + 2))
        R2 += beta0
        Sm1 += beta1
        Sm2 += beta2

    gamma = max((2 * R1 + R2) * Sp2 + Sp1 ** 2, (2 * R2 + R1) * Sm2 + Sm1 ** 2) / (N + 1)

    return R1, R2, gamma


def bern_loss_func(name, L):
    '''
    Define loss function
    input:
        name - name of loss funtion
        L - bound for prod
    output:
        loss - loss function
    '''

    if name == 'hinge':
        loss = lambda x: max(0, 1 + L - 2 * L * x)
    elif name == 'logistic':
        loss = lambda x: log(1 + exp(L - 2 * L * x))
    else:
        print('Wrong loss function!')

    return loss


def pos(N, prod, L):
    '''
    Compute positive function and gradient information
    input:
        N -
        prod - wt*xt
        L -
    output:
        fpt - positive function value
        gfpt - positive function gradient
        wasted - time wasted on computing
    '''

    plus = L / 2 + prod
    p = list(range(N + 1))
    fpt = np.power(plus, p)
    gfpt = np.multiply(fpt, p) / plus  # no xt yet!

    return fpt, gfpt


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

def SAUC(Xtr, Ytr, Xte, Yte, options, stamp=10):
    '''
    Stochastic AUC Optimization with General Loss
    input:
        Xtr - Training features
        Ytr - Training labels
        Xte - Testing features
        Yte - Testing labels
        options -
        stamp - record stamp
    output:
        elapsed_time -
        roc_auc - auc scores
    '''

    sum_time = 0.0

    wasted = 0.0

    # load parameter
    T = options['T']
    name = options['name']
    N = options['N']
    R = options['R']
    kappa = max(np.linalg.norm(Xtr,axis=1))
    print(kappa)
    L = 2*R*kappa
    c = options['c']
    B = options['B']
    sampling = options['sampling']

    # get the dimension of what we are working with
    n, d = Xtr.shape

    # initializing
    WT = np.zeros(d)
    AT = np.zeros(N + 1)
    BT = np.zeros(N + 1)
    ALPHAT = np.zeros(N + 1)

    # define loss function
    loss = bern_loss_func(name, L)

    # compute combination and coefficient
    comb_dict = comb(N)
    beta, gbeta = coef(N, loss, L, comb_dict)

    # compute gamma(get it done, bitch!)
    R1, R2, gamma = bound(N, loss, L, comb_dict)

    print('SAUC with loss = %s sampling = %s N = %d L = %d gamma = %.02f c = %d' % (name, sampling, N, L, gamma, c))

    # record auc
    roc_auc = []

    # record time elapsed
    elapsed_time = []
    start_time = time.time()

    # Begin algorithm
    for t in range(1, T + 1):

        # step size
        eta = c/sqrt(t)

        # Primal Dual Stochastic Gradient
        for j in range(t):

            wt = WT + 0.0
            at = AT + 0.0
            bt = BT + 0.0
            alphat = ALPHAT + 0.0

            Wt = wt + 0.0
            At = at + 0.0
            Bt = bt + 0.0
            ALPHAt = alphat + 0.0

            # inner production
            prod = np.dot(wt, Xtr[(t*(t-1)//2+j)%n])
            fpt, gfpt = pos(N, prod, L)
            fnt, gfnt = neg(N, prod, L, beta, gbeta)

            # if condition is faster than two inner product!
            if Ytr[(t*(t-1)//2+j)%n] == 1:
                gradwt = 2 * np.inner(alphat - at, gfpt)
                gradat = 2 * (at - fpt)
                gradbt = 2 * bt
                gradalphat = -2 * (alphat - fpt)
            else:
                gradwt = 2 * np.inner(alphat - bt, gfnt)
                gradat = 2 * at
                gradbt = 2 * (bt - fnt)
                gradalphat = -2 * (alphat - fnt)

            # update
            wt = proj(wt - eta * (gradwt * Xtr[(t*(t-1)//2+j)%n] * Ytr[(t*(t-1)//2+j)%n] / (2 * (N + 1)) + gamma * (wt - WT)),L / 2)
            at = proj(at - eta * gradat / (2 * (N + 1)),R1)
            bt = proj(bt - eta * gradbt / (2 * (N + 1)),R2)
            alphat = proj(alphat + eta * gradalphat / (2 * (N + 1)),R1+R2)

            Wt += wt
            At += at
            Bt += bt
            ALPHAt += alphat

        # update outer loop
        WT = Wt / t
        AT = At / t
        BT = Bt / t
        ALPHAT = ALPHAt / t

        elapsed_time.append(time.time() - start_time)
        roc_auc.append(roc_auc_score(Yte, np.dot(Xte, WT)))

        if t % stamp == 0:
            print('iteration: %d AUC: %.6f time eplapsed: %.2f' % (t, roc_auc[-1], elapsed_time[-1]))

    return elapsed_time, roc_auc