# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:25:13 2018

@author: Zhenhuan Yang

# -*- coding: utf-8 -*-
Spyder Editor

We apply the algorithm in Gao, 2013 ICML to do AUC maximization

Input:
    x_tr: training instances
    y_tr: training labels
    x_te: testing instances
    y_te: testing labels
    options: a dictionary 
        'ids' stores the indices of examples traversed, ids divided by # of training examples is the # of passes
        'eta' stores the initial step size
        'beta': the L2 parameter 
        'n_pass': the number of passes
        'time_lim': optional argument, the maximal time allowed
Output:
    roc_auc: results on iterates indexed by res_idx
    time:
"""
# https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r singleton 
import numpy as np
from sklearn.metrics import roc_auc_score
import time


def OPAUC(x_tr, x_te, y_tr, y_te, options):
    # options
    ids = options['ids']
    eta = options['c']
    beta = options['R']
    n, d = x_tr.shape
    # print(type(d))
    w = np.zeros(d)
    #    R = 1e7
    sp = int(0)  # the estimate of probability with positive example
    t = 0  # the time iterate"
    time_s = 0
    sx_pos = np.zeros(d)  # summation of positive instances
    sx_neg = np.zeros(d)  # summation of negative instances 
    smat_pos = np.zeros((d, d))
    smat_neg = np.zeros((d, d))
    cpt = sx_pos  # average of positive instances
    cnt = sx_neg  # average of positive instances
    # -------------------------------
    # for storing the results
    w_sum = np.zeros(d)
    eta_sum = 0
    w_sum_old = w_sum
    eta_sum_old = 0
    n_pass = options['n_pass']
    res_idx = 2 ** (np.arange(4, np.log2(n_pass * n), options['rec']))
    res_idx[-1] = n_pass * n
    res_idx = [int(i) for i in res_idx]  # map(int, res_idx)

    n_idx = len(res_idx)
    roc_auc = np.zeros(n_idx)
    elapsed_time = np.zeros(n_idx)
    i_res = 0
    # ------------------------------

    start = time.time()
    while t < len(ids):
        # print(ids[t])
        x_t = x_tr[ids[t]]
        y_t = y_tr[ids[t]]
        t += 1
        if y_t == 1:
            sp = sp + 1
            sx_pos = sx_pos + x_t
            cpt = (1 / sp) * sx_pos
            smat_pos = smat_pos + (x_t.T).dot(x_t)
            minus = x_t - cnt
            if sp == t:
                gd = 0  # beta * w + (tmp - 1) * (x_t - cnt) + np.dot(w, - (cnt.T).dot(cnt)) 
            else:
                gd = (np.inner(minus, w) - 1) * minus + np.dot(w, 1 / (t - sp) * smat_neg - (cnt.T).dot(cnt))
        else:
            sx_neg = sx_neg + x_t
            cnt = (1 / (t - sp)) * sx_neg
            smat_neg = smat_neg + (x_t.T).dot(x_t)
            minus = x_t - cpt
            if sp == 0:
                gd = 0  # beta * w + (tmp + 1) * (x_t - cpt) + np.dot(w, - (cpt.T).dot(cpt)) 
            else:
                gd = (np.inner(minus, w) + 1) * minus + np.dot(w, 1 / sp * smat_pos - (cpt.T).dot(cpt))

        # print("P=%s" % p)
        #        print(gd)    
        #        eta = eta_0 / (np.sqrt(t))
        # we use normalization
        w = w - eta * gd
        tnm = np.linalg.norm(w)
        if tnm > beta:
            w = w * (beta / tnm)
        w_sum = w_sum + w  # * eta
        eta_sum += 1  # eta
        if res_idx[i_res] == t:
            stop = time.time()
            time_s += stop - start
            w_ave = (w_sum - w_sum_old) / (eta_sum - eta_sum_old)
            pred = (x_te.dot(w_ave.T)).ravel()
            if not np.all(np.isfinite(pred)):
                print(w_ave)
                roc_auc[i_res:] = roc_auc[i_res - 1]
                elapsed_time[i_res:] = time_s
                break

            roc_auc[i_res] = roc_auc_score(y_te,pred)

            elapsed_time[i_res] = time_s

            print('iteration: %d AUC: %.6f time eplapsed: %.2f' % (t, roc_auc[i_res], elapsed_time[i_res]))

            w_sum_old = w_sum
            eta_sum_old = eta_sum
            i_res = i_res + 1

            if 'time_lim' in options and time_s > options['time_lim']:  # if slow we do not spend more time
                roc_auc[i_res:] = roc_auc[i_res - 1]
                elapsed_time[i_res:] = time_s
                break
            start = time.time()

    return elapsed_time, roc_auc