import numpy as np
from itertools import product
from matplotlib import pyplot as plt
import pickle

def compute(x):
    '''
    Compute mean and standard deviation
    '''
    BOUND = np.zeros((folders,LOSS,ALG,L,LAM,C,COMPLETE))
    MEAN = np.zeros((LOSS,ALG,L,LAM,C,COMPLETE))
    STD = np.zeros((LOSS,ALG,L,LAM,C,COMPLETE))

    for folder,loss,alg,l,lam,c,complete in product(range(folders),range(LOSS),range(ALG),range(L),range(LAM),range(C),
                                                    range(COMPLETE)):
        #BOUND[folder,loss,alg,l,lam,c,complete] = np.max(x[folder,loss,alg,l,lam,c,complete,:])
        BOUND[folder, loss, alg, l, lam, c, complete] = x[folder, loss, alg, l, lam, c, complete, -1]
    for loss,alg,l,lam,c,complete in product(range(LOSS),range(ALG),range(L),range(LAM), range(C),range(COMPLETE)):
        MEAN[loss,alg,l,lam,c,complete] = np.mean(BOUND[:,loss,alg,l,lam,c,complete])
        STD[loss,alg,l,lam, c,complete] = np.std(BOUND[:,loss,alg,l,lam,c,complete])

    print('Mean:')
    print(MEAN)
    print('Standard deviation:')
    print(STD)

    return MEAN, STD


def draw(x_dict):
    '''
    Plot AUC
    '''
    handle = ['r-','b--']
    for i,(key,value) in enumerate(x_dict.items()):
        if key[-1] == True:
            plt.plot(value[1], value[0], handle[i%len(handle)], LineWidth = 2, label=key[1]) # +r'+$\frac{\gamma}{2}||\mathbf{v} - \bar\mathbf{{v}}||^2$')
        else:
            plt.plot(value[1], value[0], handle[i%len(handle)], LineWidth = 2, label=key[1]) # + r'+$\frac{\gamma}{2}||\mathbf{w} - \bar{\mathbf{w}}||^2$')
    plt.xlabel('time')
    plt.ylabel('AUC')
    #plt.ylim(.80,.95)
    plt.legend(loc='lower right',prop={'size': 12})
    plt.show()

    return

if __name__ == '__main__':
    dataset = 'splice'

    with open('%s_table.p' % (dataset), 'rb') as table:
        x = pickle.load(table)

    with open('%s_plot.p' % (dataset), 'rb') as plot:
        x_dict = pickle.load(plot)

    folders, LOSS, ALG, L, LAM, C, COMPLETE, T = x.shape # T is general now: can be iteration or recorded auc length

    compute(x)
    draw(x_dict)