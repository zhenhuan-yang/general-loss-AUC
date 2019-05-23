'''
Stochastic AUC Optimization with General Loss by Yang et al

Author: Zhenhuan(Neyo) Yang

Date: 3/19/19
'''

import numpy as np
from math import fabs, sqrt, log, exp
import time
from sklearn.metrics import roc_auc_score

# Pre-computed combinatorial numbers
comb_dict = {0: {0: 1}, 1: {0: 1, 1: 1}, 2: {0: 1, 1: 2, 2: 1}, 3: {0: 1, 1: 3, 2: 3, 3: 1},
             4: {0: 1, 1: 4, 2: 6, 3: 4, 4: 1},
             5: {0: 1, 1: 5, 2: 10, 3: 10, 4: 5, 5: 1}, 6: {0: 1, 1: 6, 2: 15, 3: 20, 4: 15, 5: 6, 6: 1},
             7: {0: 1, 1: 7, 2: 21, 3: 35, 4: 35, 5: 21, 6: 7, 7: 1},
             8: {0: 1, 1: 8, 2: 28, 3: 56, 4: 70, 5: 56, 6: 28, 7: 8, 8: 1},
             9: {0: 1, 1: 9, 2: 36, 3: 84, 4: 126, 5: 126, 6: 84, 7: 36, 8: 9, 9: 1},
             10: {0: 1, 1: 10, 2: 45, 3: 120, 4: 210, 5: 252, 6: 210, 7: 120, 8: 45, 9: 10, 10: 1}}

def bound(N, loss, L):

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

def pos(i, prod, L):

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
    fpt = plus ** i
    gfpt = fpt * i / plus  # no xt yet!

    return fpt, gfpt


def neg(N, loss, i, prod, L):

    '''
    Compute negative function and gradient information

    input:
        N - degree of Bernstein
        loss - loss function
        i - index of function
        prod - wt*xt
        L - bound on prod

    output:
        fnt - negative function value
        gfnt - negative function gradient
    '''

    minus = L / 2 - prod
    fnt = 0.0
    gfnt = 0.0
    for k in range(i, N + 1):
        # compute forward difference
        delta = 0.0
        for j in range(k + 1):
            delta += comb_dict[k][j] * (-1) ** (k - j) * loss(j / N)

        # compute coefficient
        beta = (comb_dict[N][k] * comb_dict[k][i] * (N + 1) * delta / ((2 * L) ** k)) * (minus ** (k - i))
        # compute function value
        fnt += beta
        # compute gradient
        gfnt += beta * (k - i) / minus  # no xt yet!

    return fnt, gfnt


def w_grad(gfpt, gfnt, yt, at, bt, alphat):
    '''
    Gradient with respect to w

    input:
        fpt - positive function at t
        gfpt - positive function gradient at t
        fnt - negative function at t
        gfnt - negative function gradient at t
        yt - sample label at t
        pt - p at t
        at - a at t
        bt - b at t
        alphat - alpha at t
    output:
        gradwt - gradient w.r.t. w at t
    '''
    if yt == 1:
        gradwt = 2 * (alphat - at) * gfpt
    else:
        gradwt = 2 * (alphat - bt) * gfnt

    return gradwt


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


def a_grad(fpt, yt, at):
    '''
    Gradient with respect to a

    input:
        fpt - positive function at t
        yt - sample label at t
        pt - p at t
        at - a at t

    output:
        gradat - gradient w.r.t a at t
    '''
    gradat = 0.0
    if yt == 1:
        gradat = 2 * (at - fpt)
    else:
        gradat = 2 * at
    return gradat


def b_grad(fnt, yt, bt):
    '''
    Gradient with respect to b

    input:
        fnt - negative function at t
        yt - sample label at t
        pt - p at t
        bt - b at t

    output:
        gradbt - gradient w.r.t b at t
    '''
    gradbt = 0.0
    if yt == 1:
        gradbt = 2 * bt
    else:
        gradbt = 2 * (bt - fnt)
    return gradbt

def alpha_grad(fpt,fnt,yt,alphat):
    '''
    Gradient with respect to alpha

    input:
        fpt - positive function at t
        fnt - negative function at t
        yt - sample label at t
        alphat - alpha at t
    '''
    gradalphat = 0.0
    if yt == 1:
        gradalphat = -2*(alphat - fpt)
    else:
        gradalphat = -2*(alphat - fnt)
    return gradalphat

def prox(N, eta, loss, index, X, Y, L, R1, R2, gamma, wj, aj, bj, alphaj, bwt):
    '''
    perform proximal guided gradient descent when receive an sample

    input:
        N -
        eta - step size
        loss - loss function
        index -
        X - sample features
        Y - sample labels
        L -
        R1 -
        R2 -
        gamma -
        wj -
        aj -
        bj -
        alphaj -
        bwt -

    output:
        wj - w at jth step
        aj -
        bj -
        alphaj -
    '''

    prod = np.dot(wj, X[index])
    fpt = np.zeros(N + 1)
    gfpt = np.zeros(N + 1)
    fnt = np.zeros(N + 1)
    gfnt = np.zeros(N + 1)
    gradwt = 0.0
    gradat = np.zeros(N+1)
    gradbt = np.zeros(N+1)
    gradalphat = np.zeros(N+1)

    for i in range(N + 1):
        fpt[i], gfpt[i] = pos(i, prod, L)
        fnt[i], gfnt[i] = neg(N, loss, i, prod, L)

        gradwt += w_grad(gfpt[i], gfnt[i], Y[index], aj[i], bj[i], alphaj[i])  # accumulate i
        gradat[i] = a_grad(fpt[i], Y[index], aj[i])
        gradbt[i] = b_grad(fnt[i], Y[index], bj[i])
        gradalphat[i] = alpha_grad(fpt[i], fnt[i], Y[index], alphaj[i])

        aj[i] = aj[i] - eta * gradat[i] / (2 * (N + 1))
        bj[i] = bj[i] - eta * gradbt[i] / (2 * (N + 1))
        alphaj[i] = alphaj[i] + eta * gradalphat[i] / (2 * (N + 1))

    wj = wj - eta * (gradwt * X[index] * Y[index] / (2 * (N + 1)) + gamma * (wj - bwt))

    wj = proj(wj, L / 2)
    aj = proj(aj, R1)
    bj = proj(bj, R2)
    alphaj = proj(alphaj, R1 + R2)

    return wj, aj, bj, alphaj

def PGSPD(N, t, loss, passing_list, X, Y, L, R1, R2, gamma, c, bwt, bat, bbt, balphat):

    '''
    Proximally Guided Stochastic Primal Dual Inner loop

    input:
        N -
        t - iteration at t
        loss - loss function
        passing_list
        X -
        Y -
        L -
        R1 -
        R2 -
        gamma -
        c -
        bwt - last outer loop w
        bat - last outer loop a
        bbt - last outer loop b
        balphat - last outer loop alpha

    output:
        bwt - next outer loop w
        bat - next outer loop a
        bbt - next outer loop b
        balphat - next outer loop alpha
    '''

    # initialize inner loop variables
    Wt = bwt + 0.0
    At = bat + 0.0
    Bt = bbt + 0.0
    ALPHAt = balphat + 0.0

    BWt = 0.0
    BAt = 0.0
    BBt = 0.0
    BALPHAt = 0.0

    ETAt = c / sqrt(t) / gamma

    # inner loop update at j
    for j in range(t):
        # update inner loop variables
        Wt, At, Bt, ALPHAt = prox(N, ETAt, loss, passing_list[j], X, Y, L, R1, R2, gamma, Wt, At, Bt, ALPHAt, bwt)

        BWt += Wt
        BAt += At
        BBt += Bt
        BALPHAt += ALPHAt

    # update outer loop variables
    bwt = BWt / t
    bat = BAt / t
    bbt = BBt / t
    balphat = BALPHAt / t

    return bwt, bat, bbt, balphat

def SAUC_prev(Xtr,Xte,Ytr,Yte,options,stamp = 10):
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
    T = options['T']
    name = options['name']
    N = options['N']
    R = options['R']
    L = 2 * R * max(np.linalg.norm(Xtr, axis=1))
    c = options['c']
    B = options['B']
    sampling = options['sampling']

    # get the dimension of what we are working with
    n, d = Xtr.shape

    WT = np.zeros(d)
    AT = np.zeros(N + 1)
    BT = np.zeros(N + 1)
    ALPHAT = np.zeros(N + 1)

    # restore average wt
    avgwt = WT + 0.0

    # define loss function
    loss = bern_loss_func(name, L)

    # compute gamma
    R1, R2, gamma = bound(N,loss,L)

    print('SAUC with loss = %s sampling = %s N = %d L = %d gamma = %.02f c = %d' % (name, sampling, N, L, gamma, c))

    # record auc
    roc_auc = []

    # record time elapsed
    elapsed_time = []
    sum_time = 0.0
    start_time = time.time()

    # Begin algorithm
    for t in range(1, T + 1):

        # Prepare the indices if you know what I mean
        prep_time = time.time()
        epoch = t // n
        begin = (t * (t - 1) // 2) % n
        end = (t * (t + 1) // 2) % n
        if epoch < 1:
            if begin < end:
                tr_list = [i for i in range(begin, end)]
            else:  # need to think better
                tr_list = [i for i in range(begin, n)] + [i for i in range(end)]
        else:
            if begin < end:
                tr_list = [i for i in range(begin, n)] + [i for i in range(n)] * (epoch - 1) + [i for i in range(end)]
            else:
                tr_list = [i for i in range(begin, n)] + [i for i in range(n)] * epoch + [i for i in range(end)]
        sum_time += time.time() - prep_time
        # Inner loop
        WT, AT, BT, ALPHAT = PGSPD(N, t, loss, tr_list, Xtr, Ytr, L, R1, R2, gamma, c, WT, AT, BT, ALPHAT)
        avgwt = ((t - 1) * avgwt + WT) / t

        if t % stamp == 0:
            elapsed_time.append(time.time() - start_time - sum_time)
            roc_auc.append(roc_auc_score(Yte, np.dot(Xte, avgwt)))
            print('iteration: %d AUC: %.6f time elapsed: %.2f' % (t, roc_auc[-1], elapsed_time[-1]))

            sum_time = 0.0

    return elapsed_time, roc_auc
