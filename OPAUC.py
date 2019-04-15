'''
One-Pass AUC Optimization by Gao et al

Author: Zhenhuan(Neyo) Yang

Date: 3/21/19
'''

import numpy as np
import time
from math import sqrt
from sklearn.metrics import roc_auc_score

def OPAUC(Xtr, Ytr, Xte, Yte, options,stamp = 100):
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
    lam = options['lam']

    print('OPAUC with lambda = %.2f c  = %.2f' % (lam,c))

    # get the dimension of what we are working with
    n, d = Xtr.shape

    # initialize
    wt = np.zeros(d)
    Tpt = 0
    Tnt = 0
    cpt = np.zeros(d)
    cnt = np.zeros(d)
    Gammapt = np.zeros((d,d))
    Gammant = np.zeros((d,d))

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
            Gammapt = Gammapt + (np.outer(Xtr[t%n],Xtr[t%n]) - Gammapt)/Tpt + np.outer(cpt,cpt)
            cpt = cpt + (Xtr[t % n] - cpt) / Tpt
            Gammapt -= np.outer(cpt,cpt)
            gwt = lam*wt - Xtr[t%n] + cnt + np.inner(wt,Xtr[t%n] - cnt)*(Xtr[t%n] - cnt) + np.dot(Gammant,wt)

        else:
            Tnt += 1
            Gammant = Gammant + (np.outer(Xtr[t % n], Xtr[t % n]) - Gammant) / Tnt + np.outer(cnt, cnt)
            cnt = cnt + (Xtr[t % n] - cnt) / Tnt
            Gammant -= np.outer(cnt, cnt)
            gwt = lam * wt + Xtr[t % n] - cpt + np.inner(wt,Xtr[t % n] - cpt) * (Xtr[t % n] - cpt) + np.dot(Gammapt,wt)

        # gradient descent
        wt -= eta * gwt

        # write results
        elapsed_time.append(time.time() - start_time)
        roc_auc.append(roc_auc_score(Yte, np.dot(Xte, wt)))

        # running log
        if t % stamp == 0:
            print('iteration: %d AUC: %.6f time eplapsed: %.2f' % (t, roc_auc[-1], elapsed_time[-1]))

    return elapsed_time, roc_auc