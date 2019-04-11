'''
Stochastic AUC Optimization with General Loss by Yang et al
Author: Zhenhuan(Neyo) Yang
Date: 4/2/19
'''

import numpy as np
import time
from math import sqrt,log,floor

def gradF(prod,at,bt,alphat,pt,y):
    '''
    Compute gradient
    input:
        a -
        b -
        alpha -
        p -
        prod -
        y -
    output:
        gradwt -
        gradat -
        gradbt -
        gradalphat -
    '''
    # F = (1-pt) * (prod - at)**2 * (y+1)/2 + pt * (prod - bt)**2 * (1-y)/2 - pt * (1-pt) * alphat**2 + 2*(1+alphat) * (pt*prod*(1-y)/2 - (1-pt)*prod*(1+y)/2)

    gradwt = 2 * (1-pt) * (prod - at) * (y+1)//2 + 2 * pt * (prod - bt) * (1-y)//2 + 2*(1+alphat) * (pt*(1-y)//2 - (1-pt)*(1+y)//2) # no x yet!
    gradat = 2 * (1-pt) * (at - prod) * (y+1)//2
    gradbt = 2 * pt * (bt - prod) * (1-y)//2
    gradalphat = -2 * pt * (1-pt) * alphat + 2 * (pt*prod*(1-y)//2 - (1-pt)*prod*(1+y)//2)

    return gradwt,gradat,gradbt,gradalphat

def proj_l1(v,R):
    '''
    Efficient Projections onto the l1-Ball for Learning in High Dimensions
    Duchi et al.

    input:
        v -
        R - radius

    output:
        w -
    '''
    n = len(v)

    # initialize
    U = list(range(n))
    s = 0
    rho = 0

    # update
    while U:
        k = np.random.choice(U)
        # partition
        G = [j for j in U if v[j]>=v[k]]
        L = [j for j in U if v[j]<v[k]]

        # calculate
        delta_rho = len(G)
        delta_s = sum(v[G])

        if (s+delta_s) - (rho+delta_rho)*v[k] < R:
            s += delta_s
            p += delta_rho
            U = L + []
        else:
            U = G.pop(k)

    # set
    theta = (s-R)/rho

    # output
    w = np.maximum(v - theta,0)

    return w

def proj_l2(v,o,R):
    '''
    Projection onto eccentric l2 ball

    input:
        v -
        o - center
        R - radius

    output:
        w - projected
    '''
    norm = np.linalg.norm(v-o)
    w = (v-o) / norm * R + o

    return w


def alt_proj(v,R):
    '''
    Alternating Projection Algorithm

    input:
        v -
        R - radius

    output:
        w -
    '''
    return 

def PDSG(w,a,b,alpha,r,D,eta,R,kappa,passing_list,X,Y):

    '''
    Primal dual stochastic gradient
    input:
        w -
        a -
        b -
        alpha -
        r -
        D -
        eta -
        passing_list -
        X -
        Y -
    output:
    '''

    T = len(passing_list)
    d = len(w)
    Ap = np.zeros(d) # just d dim as last two dim is always zero
    Am = np.zeros(d)
    Tp = 0
    Tm = 0
    p = 0

    wt = w + 0.0
    at = a + 0.0
    bt = b + 0.0
    alphat = alpha + 0.0

    WT = wt + 0.0
    AT = at + 0.0
    BT = bt + 0.0
    # ALPHAT = alphat + 0.0

    for t in passing_list:

        # update
        Ap += (1+Y[t])//2 * X[t]
        Am += (1-Y[t])//2 * X[t]
        Tp += (1+Y[t])//2
        Tm += (1-Y[t])//2
        p = Tp / (Tp + Tm)

        # gradient
        gradwt,gradat,gradbt,gradalphat = gradF(prod,at,bt,alphatpt,p,Y[t])
        wt -= eta * gradwt * X[t]
        at -= eta * gradat
        bt -= bta * gradbt
        alphat += eta * gradalphat

        # projection
        wt = proj(wt)
        at = proj(at)
        bt = proj(bt)
        alphat = proj(alphat)

        # accumulate
        WT += wt
        AT += at
        BT += bt
        # ALPHAT += alphat

    # compute
    WT = WT/T
    AT = AT/T
    BT = BT/T
    ALPHAT = np.inner(Am/Tm - Ap/Tp,WT)

    # lemma
    r = r/2
    D = 2*sqrt(2)*kappa*r + 4*sqrt(2)*kappa*(2+sqrt(2*log(12/delta)))*(1+2*kappa)*R / sqrt((min(p,1-p)*T - sqrt(2*T*log(12/delta))))
    beta = (1+8*kappa**2) + 32*kappa**2*(1+2*kappa)**2*(2+sqrt(2*log(12/delta)))**2 / (min(p,1-p) - sqrt(2*log(12/delta)/T))

    return WT,AT,BT,ALPHAT,beta,r,D

def FSAUC(n,R,c,Xtr,Ytr,Xte,Yte,stamp = 10):
    '''
    Fast Stochastic AUC Maximization
    input:
        n - total iteration
        R -
        c - step size
        Xtr -
        Ytr -
        Xte -
        Yte -
        stamp - record stamp
    output:
        Wt -
    '''
    # normalized data
    kappa = 1

    # get the dimension of what we are working with
    N, d = Xtr.shape

    # set
    log2 = lambda x: log(x)/log(2)
    m = floor(log2(2*n/log2(n))/2) - 1
    n0 = floor(n/m)
    r = 2*sqrt(1+2*kappa**2)*R # R0
    G = max((1+4*kappa)*kappa*(R+1),2*kappa*(2*R+1+2*R*kappa),2*kappa*(4*kappa*R+11*R+1)) * c
    beta = (1+8*kappa**2)
    D = 2*sqrt(2)*kappa*r

    # initialize
    Wt = np.zeros(d)
    At = np.zeros(1)
    Bt = np.zeros(1)
    ALPHAt = np.zeros(1)

    # record auc
    roc_auc = []

    # record time elapsed
    elapsed_time = []
    sum_time = 0.0
    start_time = time.time()

    for k in range(m):
        # prepare
        prep_time = time.time()
        epoch = n0 // N
        begin = k * n0 // N
        end = (k+1) * n0 // N
        if epoch < 1:
            if begin < end:
                tr_list = [i for i in range(begin, end)]
            else:
                tr_list = [i for i in range(begin, N)] + [i for i in range(end)]
        else:
            if begin < end:
                tr_list = [i for i in range(begin, N)] + [i for i in range(N)] * (epoch - 1) + [i for i in range(end)]
            else:
                tr_list = [i for i in range(begin, N)] + [i for i in range(N)] * epoch + [i for i in range(end)]
        sum_time += time.time() - prep_time

        eta = sqrt(beta)/(sqrt(3*n0)*G) * r
        # inner loop
        Wt, At, Bt, ALPHAt, beta, r, D = PDSG(Wt,At,Bt,ALPHAt,r,D,eta,R,kappa,tr_list,Xtr,Ytr)

        if t % stamp == 0:
            elapsed_time.append(time.time() - start_time - sum_time)
            roc_auc.append(roc_auc_score(Yte, np.dot(Xte, Wt)))
            print('R: %.2f c: %.2f iteration: %d AUC: %.6f time eplapsed: %.2f' % (R, c, t, roc_auc[-1], elapsed_time[-1]))

            sum_time = 0.0

    return elapsed_time, roc_auc
