"""
Created on Fri Nov 16 15:53:25 2018
@author: Zhenhuan Yang
We apply the algorithm in Zhao, 2011 ICML to do AUC maximization
Input:
    x_tr: training instances
    y_tr: training labels
    x_te: testing instances
    y_te: testing labels
    options: a dictionary
        'ids' stores the indices of examples traversed, ids divided by number of training examples is the number of passes
        'eta' the parameter C in the paper
        'n_pass': the number of passes
        'time_lim': optional argument, the maximal time allowed
        'Np': the buffer size of positive label
        'Nn': the buffer size of negative label
Output:
    roc_auc: results on iterates indexed by res_idx
    time:
"""
import numpy as np
from sklearn.metrics import roc_auc_score
import time


def OAM(x_tr, x_te, y_tr, y_te, options):
    # options
    ids = options['ids']
    R = options['R']
    c = options['c']
    T = len(ids)

    # initialization
    n, d = x_tr.shape
    wt = np.zeros(d)
    t = 0  # the time iterate
    time_s = 0
    Np, Nn = options['Np'], options['Nn']
    Bp = np.zeros([Np, d]) # just store it
    Bn = np.zeros([Nn, d])
    n_pos, n_neg = 0, 0

    # -------------------------------
    # for storing the results
    w_sum = np.zeros(d)
    eta_sum = 0
    w_sum_old = w_sum
    eta_sum_old = 0
    n_pass = options['n_pass']
    res_idx = 2 ** (np.arange(4, np.log2(n_pass * n), options['rec']))
    res_idx[-1] = n_pass * n  # make sure the last step recorded
    res_idx = [1]+[int(i) for i in res_idx]  # map(int, res_idx)

    n_idx = len(res_idx)
    roc_auc = np.zeros(n_idx)
    elapsed_time = np.zeros(n_idx)
    i_res = 0
    # ------------------------------
    print('OAM with R = %d c = %d Np = %d Nn = %d' % (R, c, Np, Nn))

    start = time.time()
    while t < T:
        # print(ids[t])
        x_t = x_tr[ids[t]]
        y_t = y_tr[ids[t]]

        if y_t == 1:
            n_pos += 1
            ct = c * max(1, n_neg / Nn)
            if n_pos <= Np:
                Bp[n_pos - 1] = x_t
            else:
                Bp = reservoir(Bp, x_t, Np, n_pos)
            xx = x_t - Bn
            idx = ((xx @ wt) <= 1)
            wt = wt + ct * np.sum(xx[idx, :], axis=0) / 2
        else:
            n_neg += 1
            ct = c * max(1, n_pos / Np)
            if n_neg <= Nn:
                Bn[n_neg - 1] = x_t
            else:
                Bn = reservoir(Bn, x_t, Nn, n_neg)
            diff = x_t - Bp
            idx = ((diff @ wt) >= -1)
            wt = wt - ct * np.sum(diff[idx, :], axis=0) / 2

        tnm = np.linalg.norm(wt)
        if tnm > R:
            wt = wt * (R / tnm)

        t = t + 1
        w_sum = w_sum + wt
        eta_sum += 1

        if res_idx[i_res] == t:
            stop = time.time()
            time_s += stop - start
            w_ave = (w_sum - w_sum_old) / (eta_sum - eta_sum_old)
            pred = x_te @ w_ave
            if not np.all(np.isfinite(pred)):
                break

            roc_auc[i_res] = roc_auc_score(y_te, pred)
            elapsed_time[i_res] = time_s

            print('iteration: %d AUC: %.6f time eplapsed: %.2f' % (t, roc_auc[i_res], elapsed_time[i_res]))
            w_sum_old = w_sum
            eta_sum_old = eta_sum
            i_res += 1

            start = time.time()

    return elapsed_time, roc_auc


def reservoir(buf, x, buf_size, n):
    rd = np.random.rand(1)
    if rd[0] < buf_size / n:
        idx = np.random.randint(buf_size, size=1)
        buf[idx[0]] = x

    return buf