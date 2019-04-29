import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

if __name__ == '__main__':

    alg = 'SAUC'
    dataset = 'pendigits'

    # Read
    df = pd.read_pickle('/home/neyo/PycharmProjects/AUC/results/cv_%s_%s.h5' % (alg,dataset))

    R = [.01, .1, 1, 10, 100]
    C = [.01, .1, 1, 10, 100]

    result = pd.DataFrame()
    ind = pd.DataFrame()
    # Results
    for c,r in product(C,R):
        result[(c, r)] = [np.max(df[(c, r)]['MEAN'])]


        plt.plot(df[(c, r)]['MEAN'], label='c= %.2f R = %.2f AUC = %.4f' % (c, r, result[(c,r)]))


    column_ind = np.argmax(result.values)
    column = result.columns[column_ind]
    ind = np.argmax(df[column]['MEAN'])
    print('alg = %s data = %s c = %.2f R = %.2f AUC = ' % (alg, dataset, column[0], column[1]), end = ' ')
    print(('%.4f$\pm$'%result[column]).lstrip('0'), end = '')
    print(('%.4f'%df[column]['STD'][ind]).lstrip('0'))
    plt.xlabel('Iteration')
    plt.ylabel('AUC')
    plt.ylim([.5, 1])
    plt.legend(loc=4)
    plt.title('%s_%s' % (alg, dataset))
    plt.show()
