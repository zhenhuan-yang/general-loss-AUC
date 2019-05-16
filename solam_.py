"""
Created on Tue May  6 09:25:13 2019
@author: Zhenhuan Yang
# -*- coding: utf-8 -*-
Spyder Editor
We apply the algorithm in Ying, 2016 NIPS to do AUC maximization
Input:
    x_tr: training instances
    y_tr: training labels
    x_te: testing instances
    y_te: testing labels
    options: a dictionary
        'ids' stores the indices of examples traversed, ids divided by number of training examples is the number of passes
        'eta' stores the initial step size
        'R': the parameter R or the L-2 regularizer (depending on the algorithm)
        'n_pass': the number of passes
        'time_lim': optional argument, the maximal time allowed
Output:
    aucs: results on iterates indexed by res_idx
    time:
auc_solam_L2 is the original implementation in Ying 2016, NIPS
auc_solam_L2: is the variant with stochastic proximal AUC maximization, L2 regularizer is processed analogously by natole 2018 ICML
"""
# https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r singleton
import numpy as np
from sklearn.metrics import roc_auc_score
import time

def SOLAM(x_tr, x_te, y_tr, y_te, options):

    # options
    ids = options['ids']
    c = options['c']
    R = options['R']
    T = len(ids)
    series = np.arange(1, T + 1, 1)
    etas = c / (np.sqrt(series)) # define eta outside makes it a little faster
    n, d = x_tr.shape
    wt = np.zeros(d)
    at = 0
    bt = 0
    alphat = 0
    sp = 0  # the estimate of probability with positive example
    time_s = 0
    t = 0  # the time iterate"
    # -------------------------------
    # for storing the results
    bwt = np.zeros(d)
    beta = 0
    bwt_old = bwt + 0
    beta_old = 0

    n_pass = options['n_pass']
    res_idx = 2 ** (np.arange(4, np.log2(n_pass * n), options['rec']))
    res_idx[-1] = n_pass * n
    res_idx = [int(i) for i in res_idx]  # map(int, res_idx)
    n_idx = len(res_idx)
    roc_auc = np.zeros(n_idx)
    elapsed_time = np.zeros(n_idx)
    i_res = 0
    # ------------------------------
    print('SOLAM with R = %d c = %d' % (R, c))
    start = time.time()
    while t < T:

        x_t = x_tr[ids[t]]
        y_t = y_tr[ids[t]]
        prod = np.inner(wt, x_t)  # np.inner(x_t, v[:dim])
        eta = etas[t]

        t = t + 1

        if y_t == 1:
            sp = sp + 1
            p = sp / t
            gradwt = (1 - p) * (prod - at - 1 - alphat) * x_t
            gradat = (p - 1) * (prod - at)
            gradbt = 0
            gradalphat = (p - 1) * (prod + p * alphat)
        else:
            p = sp / t
            gradwt = p * (prod - bt + 1 + alphat) * x_t
            gradat = 0
            gradbt = p * (bt - prod)
            gradalphat = p * (prod + (p - 1) * alphat)
        wt = wt - eta * gradwt
        at = at - eta * gradat
        bt = bt - eta * gradbt
        alphat = alphat + eta * gradalphat

        # some projection
        # ---------------------------------
        tnm = np.linalg.norm(wt)
        if tnm > R:
            wt = wt * (R / tnm)
        tnm = np.abs(at)
        if tnm > R:
            at = at * (R / tnm)
        tnm = np.abs(bt)
        if tnm > R:
            bt = bt * (R / tnm)
        tnm = np.abs(alphat)
        if tnm > 2 * R:
            alphat = alphat * (2 * R / tnm)
        # ---------------------------------

        # update output
        bwt += wt * eta

        beta += eta

        if res_idx[i_res] == t:

            stop = time.time()
            time_s += stop - start
            bw = (bwt - bwt_old) / (beta - beta_old)

            pred = x_te @ bw # returns flat array
            if not np.all(np.isfinite(pred)):
                break

            roc_auc[i_res] = roc_auc_score(y_te, pred)

            elapsed_time[i_res] = time_s

            print('iteration: %d AUC: %.6f time eplapsed: %.2f' % (t, roc_auc[i_res], elapsed_time[i_res]))
            bwt_old = bwt + 0
            beta_old = beta + 0
            i_res += 1
            start = time.time()

    return elapsed_time, roc_auc