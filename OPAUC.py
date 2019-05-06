'''
One-Pass AUC Optimization by Gao et al
Author: Zhenhuan(Neyo) Yang
Date: 3/21/19
'''

import numpy as np
import time
from math import sqrt
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

def OPAUC(Xtr,Xte,Ytr,Yte, options,stamp = 10):
    '''
    One-Pass AUC Optimization
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
    R = options['R'] # modified algorithm to be bounded not regularized
    tau = options['tau']

    # get the dimension of what we are working with
    n, d = Xtr.shape

    # choose to approximate or not
    if d<tau:
        print('OPAUC with R = %.2f c  = %.2f dimension = %d' % (R, c, d))
        cov = 'full'
    else:
        print('OPAUC with R = %.2f c  = %.2f dimension = %d tau = %d' %(R,c,d,tau))
        cov = 'approximate'

    # initialize
    wt = np.zeros(d)
    Tpt = 0
    Tnt = 0
    cpt = np.zeros(d)
    cnt = np.zeros(d)
    if cov == 'full':
        Gammapt = np.zeros((d,d))
        Gammant = np.zeros((d,d))
    elif cov == 'approximate':
        Gammapt = np.zeros((d,tau))
        Gammant = np.zeros((d,tau))
        Rpt = np.zeros(tau) # record accumulative gaussian vectors
        Rnt = np.zeros(tau)
        # store for convenience
        cpt_hat = np.zeros((d,tau))
        cnt_hat = np.zeros((d,tau))
        Spt_hat = np.zeros((d,d))
        Snt_hat = np.zeros((d,d))
    else:
        print('Wrong covariance option!')
        return

    # record auc
    roc_auc = []

    # record time elapsed
    elapsed_time = []
    start_time = time.time()
    for t in range(1,T+1):

        # step size
        eta = c / sqrt(t)

        if Ytr[t % n] == 1:
            Tpt += 1
            if cov == 'full':
                temp = cpt + 0.0 #
                cpt = cpt + (Xtr[t % n] - cpt) / Tpt
                Gammapt = Gammapt + (np.outer(Xtr[t%n], Xtr[t%n]) - Gammapt)/Tpt + np.outer(temp,temp) - np.outer(cpt,cpt)
                gwt = -Xtr[t%n] + cnt + (np.outer(Xtr[t%n] - cnt, Xtr[t%n] - cnt) + Gammant)@wt
            elif cov == 'approximate':
                rt = np.random.randn(tau)
                Rpt += rt/tau
                Gammapt = Gammapt + np.outer(Xtr[t % n], rt) / sqrt(tau)  # note there is typo in icml version
                Spt_hat = Gammapt @ Gammapt.transpose() / Tpt - cpt_hat @ cpt_hat.transpose()
                cpt = cpt + (Xtr[t % n] - cpt) / Tpt

                cpt_hat = np.outer(cpt,Rpt)/Tpt
                gwt = -Xtr[t%n] + cnt + (np.outer(Xtr[t%n] - cnt, Xtr[t%n] - cnt) + Snt_hat)@wt
            else:
                print('Wrong covariance option!')
                return
        else:
            Tnt += 1
            if cov == 'full':
                temp = cnt + 0.0
                cnt = cnt + (Xtr[t % n] - cnt) / Tnt
                Gammant = Gammant + (np.outer(Xtr[t % n], Xtr[t % n]) - Gammant) / Tnt + np.outer(temp, temp) - np.outer(cnt, cnt)
                gwt = Xtr[t % n] - cpt + (np.outer(Xtr[t % n] - cpt, Xtr[t % n] - cpt) + Gammapt) @ wt
            elif cov == 'approximate':
                rt = np.random.randn(tau)
                Rnt += rt/tau
                Gammant = Gammant + np.outer(Xtr[t % n], rt) / sqrt(tau)  # note there is typo in icml version
                Snt_hat = Gammant @ Gammant.transpose() / Tnt - cnt_hat @ cnt_hat.transpose()
                cnt = cnt + (Xtr[t % n] - cnt) / Tnt

                cnt_hat = np.outer(cnt, Rnt) / Tnt
                gwt = Xtr[t % n] - cpt + (np.outer(Xtr[t % n] - cpt, Xtr[t % n] - cpt) + Spt_hat) @ wt
            else:
                print('Wrong covariance option!')
                return

        # gradient descent
        wt = proj(wt - eta * gwt, R)

        # write results
        elapsed_time.append(time.time() - start_time)
        roc_auc.append(roc_auc_score(Yte, np.dot(Xte, wt)))

        # running log
        if t % stamp == 0:
            print('iteration: %d AUC: %.6f time eplapsed: %.2f' % (t, roc_auc[-1], elapsed_time[-1]))

    return elapsed_time, roc_auc