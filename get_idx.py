"""
Created on Thu Mar 14 09:21:35 2019
@author: Zhenhuan Yang
"""

"""
This function calculates the indices for SGD
Input:
    n_data: number of training examples
    n_pass: number of passes

Output:
    idx: the indices
"""
import numpy as np


def get_idx(n_data, n_pass):
    idx = np.zeros(n_data * n_pass, dtype=int)
    for i_pass in np.arange(n_pass):
        idx[i_pass * n_data: (i_pass + 1) * n_data] = np.arange(n_data)


    return idx