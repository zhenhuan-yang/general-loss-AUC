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

    start = time.time()

    plus = L / 2 + prod
    p = list(range(N + 1))
    fpt = np.power(plus, p)
    gfpt = np.multiply(fpt, p) / plus  # no xt yet!

    wasted = time.time() - start

    return fpt, gfpt, wasted


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

    start = time.time()

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

    wasted = time.time() - start

    return fnt, gfnt, wasted


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

def reservoir(tr_list,t,B,M):
    '''
    Reservior sampling

    input:
        tr_list - training list mask
        t - current index of sample
        B - buffer size
        M - total number of samples

    output:
        tr_list - updated training list mask
    '''
    if len(tr_list) < B:
        tr_list.append(t%M)
    else:
        z = np.random.binomial(1, p= B/t)
        if z == 1:
            ind = np.random.randint(len(tr_list))
            tr_list[ind] = t%M

    return tr_list

def sequential(t,M):
    '''
    Sequential sampling

    input:
        t - current sample
        M - number of total sample size

    output:
        tr_list - training list mask
    '''

    epoch = t // M
    begin = (t * (t - 1) // 2) % M
    end = (t * (t + 1) // 2) % M

    if epoch < 1:
        if begin < end:
            tr_list = [i for i in range(begin, end)]
        else:  # need to think better
            tr_list = [i for i in range(begin, M)] + [i for i in range(end)]
    else:
        if begin < end:
            tr_list = [i for i in range(begin, M)] + [i for i in range(M)] * (epoch - 1) + [i for i in range(end)]
        else:
            tr_list = [i for i in range(begin, M)] + [i for i in range(M)] * epoch + [i for i in range(end)]

    return tr_list


def prox(N, eta, loss, index, X, Y, L, R1, R2, gamma, beta, gbeta, wj, aj, bj, alphaj, bwt):
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
        gamma - weakly convex coefficient
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
        wasted - time wasted on computing
    '''

    prod = np.dot(wj, X[index])
    wasted = 0.0

    fpt, gfpt, _ = pos(N, prod, L)
    wasted += _
    fnt, gfnt, _ = neg(N, prod, L, beta, gbeta)
    wasted += _

    # if condition is faster than two inner product!
    if Y[index] == 1:
        gradwt = 2 * np.inner(alphaj - aj, gfpt)
        gradat = 2 * (aj - fpt)
        gradbt = 2 * bj
        gradalphat = -2 * (alphaj - fpt)
    else:
        gradwt = 2 * np.inner(alphaj - bj, gfnt)
        gradat = 2 * aj
        gradbt = 2 * (bj - fnt)
        gradalphat = -2 * (alphaj - fnt)

    wj = wj - eta * (gradwt * X[index] * Y[index] / (2 * (N + 1)) + gamma * (wj - bwt))
    aj = aj - eta * gradat / (2 * (N + 1))
    bj = bj - eta * gradbt / (2 * (N + 1))
    alphaj = alphaj + eta * gradalphat / (2 * (N + 1))

    wj = proj(wj, L / 2)
    aj = proj(aj, R1)
    bj = proj(bj, R2)
    alphaj = proj(alphaj, R1 + R2)

    return wj, aj, bj, alphaj, wasted


def PGSPD(N, t, loss, passing_list, X, Y, L, R1, R2, gamma, c, beta, gbeta, bwt, bat, bbt, balphat):
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
        wasted - time wasted on computing
    '''

    wasted = 0.0
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
        Wt, At, Bt, ALPHAt, _ = prox(N, ETAt, loss, passing_list[j], X, Y, L, R1, R2, gamma, beta, gbeta, Wt, At, Bt,
                                     ALPHAt, bwt)
        wasted += _

        BWt += Wt
        BAt += At
        BBt += Bt
        BALPHAt += ALPHAt

    # update outer loop variables
    bwt = BWt / t
    bat = BAt / t
    bbt = BBt / t
    balphat = BALPHAt / t

    return bwt, bat, bbt, balphat, wasted


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
    L = options['L']
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

    # restore average WT
    avgWT = WT + 0.0

    # define loss function
    loss = bern_loss_func(name, L)

    # compute combination and coefficient
    comb_dict = comb(N)
    beta, gbeta = coef(N, loss, L, comb_dict)

    # compute gamma(get it done, bitch!)
    R1, R2, gamma = bound(N, loss, L, comb_dict)

    print('SAUC with loss = %s N = %d L = %d gamma = %.02f c = %d' % (name, N, L, gamma, c))

    # record training list mask
    tr_list = []

    # record auc
    roc_auc = []

    # record time elapsed
    elapsed_time = []
    start_time = time.time()

    # Begin algorithm
    for t in range(1, T + 1):

        # Prepare the indices and count preparing time
        prep_time = time.time()
        if sampling == 'sequential':
            tr_list = sequential(t,n)
        elif sampling == 'reservoir':
            tr_list = reservoir(tr_list,t,B,t)
            print(tr_list)
        else:
            print('Wrong sampling option!')
            return
        sum_time += time.time() - prep_time
        # Inner loop
        WT, AT, BT, ALPHAT, _ = PGSPD(N, t, loss, tr_list, Xtr, Ytr, L, R1, R2, gamma, c, beta, gbeta, WT, AT, BT,
                                      ALPHAT)
        elapsed_time.append(time.time() - start_time - sum_time)
        wasted += _
        avgWT = ((t-1)*avgWT +WT) / t
        roc_auc.append(roc_auc_score(Yte, np.dot(Xte, avgWT)))

        if t % stamp == 0:
            print('iteration: %d AUC: %.6f time eplapsed: %.2f/%.2f' % (t, roc_auc[-1], elapsed_time[-1], wasted))

    return elapsed_time, roc_auc