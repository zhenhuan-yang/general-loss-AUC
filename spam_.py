"""
Created on Tue May  6 09:25:13 2019
@author: Zhenhuan Yang
# -*- coding: utf-8 -*-
Spyder Editor
We apply the algorithm in Natole, 2018 ICML to do AUC maximization
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
    roc_auc: results on iterates indexed by res_idx
    elapsed_time:
"""
import numpy as np
from sklearn.metrics import roc_auc_score
import time

# for this algorithm, beta is the L-2 parameter and the algorithm is the stochastic proximal AUC maximization with the L-2 regularizer
def SPAM(x_tr, x_te, y_tr, y_te, options):

    # options
    ids = options['ids']
    c = options['c']
    R = options['R']  # beta is the parameter R, we use beta for consistency
    T = len(ids)
    series = np.arange(1, T + 1, 1)
    etas = c / (np.sqrt(series))
    n, d = x_tr.shape
    wt = np.zeros(d)

    p = np.sum(y_tr[y_tr == 1]) / n # the estimate of probability with positive example
    mpt = np.mean(x_tr[y_tr == 1], axis=0)
    mnt = np.mean(x_tr[y_tr == -1], axis=0)

    time_s = 0
    t = 0  # the time iterate"
    # -------------------------------
    # for storing the results
    avgwt = np.zeros(d)

    n_pass = options['n_pass']
    res_idx = 2 ** (np.arange(4, np.log2(n_pass * n), options['rec']))
    res_idx[-1] = n_pass * n
    res_idx = [int(i) for i in res_idx]  # map(int, res_idx)
    n_idx = len(res_idx)
    roc_auc = np.zeros(n_idx)
    elapsed_time = np.zeros(n_idx)
    i_res = 0
    # ------------------------------
    print('SPAM with R = %d c = %d' % (R, c))
    start = time.time()

    while t < T:

        x_t = x_tr[ids[t]]
        y_t = y_tr[ids[t]]

        prod = np.inner(wt, x_t)
        at = np.inner(wt, mpt)
        bt = np.inner(wt, mnt)
        alphat = at - bt

        eta = etas[t]
        t = t + 1
        if y_t == 1:
            gradwt = 2 * (1 - p) * (prod - at) - 2 * (1 + alphat) * (1 - p)
        else:
            gradwt = 2 * p * (prod - bt) + 2 * (1 + alphat) * p

        wt = wt - eta * gradwt * x_t

        tnm = np.linalg.norm(wt)
        if tnm > R:
            wt = wt * (R / tnm)

        avgwt = ((t - 1) * avgwt + wt) / t

        if res_idx[i_res] == t:

            stop = time.time()
            time_s += stop - start

            pred = x_te @ avgwt  # returns flat array
            if not np.all(np.isfinite(pred)):
                break

            roc_auc[i_res] = roc_auc_score(y_te, pred)
            elapsed_time[i_res] = time_s

            print('iteration: %d AUC: %.6f time eplapsed: %.2f' % (t, roc_auc[i_res], elapsed_time[i_res]))
            i_res += 1
            start = time.time()

    return elapsed_time, roc_auc