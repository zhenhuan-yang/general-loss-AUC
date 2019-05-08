# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:25:13 2018
@author: Zhenhuan Yang
# -*- coding: utf-8 -*-
Spyder Editor
We apply the algorithm in Liu, 2018 ICML to do Fast AUC maximization
Input:
    x_tr: training instances
    y_tr: training labels
    x_te: testing instances
    y_te: testing labels
    options: a dictionary
        'ids' stores the indices of examples traversed, ids divided by number of training examples is the number of passes
        'c' stores the initial step size
        'beta': the parameter R
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


def FSAUC(x_tr, x_te, y_tr, y_te, options):
    # options
    delta = options['delta']
    dd = np.log(12 / delta)  # a term in D and beta
    ids = options['ids']
    n_ids = len(ids)
    c = options['c']
    R = options['R']  # beta is the parameter R, we use beta for consistency
    n, d = x_tr.shape
    v_1, alpha_1 = np.zeros(d + 2), 0
    sp = 0  # the estimate of probability with positive example
    t = 0  # the time iterate"
    time_s = 0
    Ap = np.zeros(d)  # summation of positive instances
    An = np.zeros(d)  # summation of negative instances
    m_pos = Ap
    m_neg = An
    # -------------------------------
    # for storing the results
    n_pass = options['n_pass']
    res_idx = 2 ** (np.arange(4, np.log2(n_pass * n), options['rec']))

    n_idx = len(res_idx)
    roc_auc = np.zeros(n_idx)
    elapsed_time = np.zeros(n_idx)
    i_res = 0
    # ------------------------------

    # we have normalized the data
    m = int(0.5 * np.log2(2 * n_ids / np.log2(n_ids))) - 1
    n0 = int(n_ids / m)
    r = 2 * np.sqrt(3) * R  # R0
    # G = max(5 * (R + 1), 2  * (4 * R + 1), 2 * (15 * R + 1)) * c
    beta = 9
    D = 2 * np.sqrt(2) * r

    res_idx[-1] = n0 * m
    res_idx = [int(i) for i in res_idx]  # map(int, res_idx)
    # pred = y_te
    gradvt = np.zeros(d + 2)

    start = time.time()

    # loop iterates k and kk are independent of data iterate t
    # as data iterate is preassigned by get_idx
    for k in range(m):

        v_sum = np.zeros(d + 2)
        v, alpha = v_1, alpha_1
        for kk in range(n0):
            x_t = x_tr[ids[t]]
            y_t = y_tr[ids[t]]
            prod = np.inner(x_t, v[:d])
            if y_t == 1:
                sp = sp + 1
                p = sp / (t + 1)  # not while loop
                Ap = Ap + x_t
                gradvt[:d] = (1 - p) * (prod - v[d] - 1 - alpha) * x_t
                gradvt[d] = (p - 1) * (prod - v[d])
                gradvt[d + 1] = 0
                gradalphat = (p - 1) * (prod + p * alpha)
            else:
                p = sp / (t + 1)
                An = An + x_t
                gradvt[:d] = p * (prod - v[d + 1] + 1 + alpha) * x_t
                gradvt[d] = 0
                gradvt[d + 1] = p * (v[d + 1] - prod)
                gradalphat = p * (prod + (p - 1) * alpha)

            t = t + 1
            v = v - c * gradvt
            alpha = alpha + c * gradalphat

            # some projection
            # ---------------------------------
            v[:d] = ProjectOntoL1Ball(v[:d], R)
            tnm = np.abs(v[d])
            if tnm > R:
                v[d] = v[d] * (R / tnm)
            tnm = np.abs(v[d + 1])
            if tnm > R:
                v[d + 1] = v[d + 1] * (R / tnm)
            tnm = np.abs(alpha)
            if tnm > 2 * R:
                alpha = alpha * (2 * R / tnm)

            vd = v - v_1
            tnm = np.linalg.norm(vd)
            if tnm > r:
                vd = vd * (r / tnm)
            v = v_1 + vd
            ad = alpha - alpha_1
            tnm = np.abs(ad)
            if tnm > D:
                ad = ad * (D / tnm)
            alpha = alpha_1 + ad
            # ---------------------------------

            v_sum = v_sum + v
            v_ave = v_sum / (t + 1)
            if res_idx[i_res] == t:
                stop = time.time()
                time_s += stop - start

                pred = x_te @ v_ave[:d]
                if not np.all(np.isfinite(pred)):
                    break
                roc_auc[i_res] = roc_auc_score(y_te, pred)
                elapsed_time[i_res] = time_s

                print('iteration: %d AUC: %.6f time eplapsed: %.2f' % (t, roc_auc[i_res], elapsed_time[i_res]))

                i_res += 1

                start = time.time()
        if not np.all(np.isfinite(pred)):
            break
        r = r / 2
        # update D and beta
        tmp1 = 12 * np.sqrt(2) * (2 + np.sqrt(2 * dd)) * R
        tmp2 = min(p, 1 - p) * n0 - np.sqrt(2 * n0 * dd)
        if tmp2 > 0:
            D = 2 * np.sqrt(2) * r + tmp1 / np.sqrt(tmp2)
        else:
            D = 1e7
        tmp1 = 288 * ((2 + np.sqrt(2 * dd)) ** 2)
        tmp2 = min(p, 1 - p) - np.sqrt(2 * dd / n0)
        if tmp2 > 0:
            beta_new = 9 + tmp1 / tmp2
        else:
            beta_new = 1e7
        c = min(np.sqrt(beta_new / beta) * c / 2, c)
        beta = beta_new
        if sp > 0:
            m_pos = Ap / sp
        if sp < t:
            m_neg = An / (t - sp)
        v_1 = v_ave
        alpha_1 = np.inner(m_neg - m_pos, v_ave[:d])

    return elapsed_time, roc_auc


def ProjectOntoL1Ball(v, b):
    nm = np.abs(v)
    if nm.sum() < b:
        w = v
    else:
        u = np.sort(nm)[::-1]
        sv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(v) + 1) > (sv - b))[0][-1]
        thc = (sv[rho] - b) / (rho + 1)
        w = np.sign(v) * np.maximum(nm - thc, 0)
    return w