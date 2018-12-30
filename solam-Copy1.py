#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import h5py
from sklearn.metrics import roc_auc_score
import time
from random import shuffle
from itertools import product
import multiprocessing as mp
from math import sqrt,log,exp,fabs
# In[2]:


def pos(i,prod,L):
    '''
    Compute positive function and gradient information
    
    input:
        i - index of function
        t - iteration
        prod - wt*xt
        
    output:
        fpt - positive function value
        gfpt - positive function gradient
    '''
    plus = L/2+prod
    fpt = plus**i
    gfpt = fpt*i/plus # no xt yet!
    
    # hfpt = i*(i-1)*(L/2+prod)**(i-2)
    return fpt,gfpt              


# In[3]:


def comb(n, k):
    '''
    Compute combination
    
    input:
        n - total number
        k - number of chosen
    
    output:
        c - number of combination
    '''
    return factorial(n) / factorial(k) / factorial(n - k)


# In[4]:


comb_dict = {0:{0:1},1:{0:1,1:1},2:{0:1,1:2,2:1},3:{0:1,1:3,2:3,3:1},4:{0:1,1:4,2:6,3:4,4:1},
             5:{0:1,1:5,2:10,3:10,4:5,5:1},6:{0:1,1:6,2:15,3:20,4:15,5:6,6:1},
             7:{0:1,1:7,2:21,3:35,4:35,5:21,6:7,7:1},8:{0:1,1:8,2:28,3:56,4:70,5:56,6:28,7:8,8:1},
             9:{0:1,1:9,2:36,3:84,4:126,5:126,6:84,7:36,8:9,9:1},
             10:{0:1,1:10,2:45,3:120,4:210,5:252,6:210,7:120,8:45,9:10,10:1}}


# In[5]:


beta_dict = {0:np.array([11,110,495,1320,2310,2772,2310,1320,495,110,11]),
             1:np.array([110,990,3960,9240,13860,13860,9240,3960,990,110]),
             2:np.array([495,3960,13860,27720,34650,27720,13860,3960,495]),
             3:np.array([1320,9240,27720,46200,46200,27720,9240,1320]),
             4:np.array([2310,13860,34650,46200,34650,13860,2310]),5:np.array([2772,13860,27720,27720,13860,2772]),
             6:np.array([2310,9240,13860,9240,2310]),7:np.array([1320,3960,3960,1320]),8:np.array([495,990,495]),
             9:np.array([110,110]),10:np.array([11])}


# In[6]:


def neg(loss,i,prod,L):
    '''
    Compute negative function and gradient information
    
    input:
        loss - loss function
        i - index of function
        t - iteration
        prod - wt*xt
        
    output:
        fnt - negative function value
        gfnt - negative function gradient
    '''

    minus = L/2-prod
    #FNT = np.zeros(N+1-i)
    #GFNT = np.zeros(N+1-i)
    fnt = 0.0
    gfnt = 0.0
    # hfnt = 0.0
    run_time = time.time()
    for k in range(i,N+1):
        # compute forward difference
        delta = 0.0
        for j in range(k+1):
            delta += comb_dict[k][j]*(-1)**(k-j)*loss(j/N)
            
        # compute coefficient
        #beta = beta_dict[i][k-i]*delta/(2*L)**k*minus**(k-i)
        beta = comb_dict[N][k]*comb_dict[k][i]*(N+1)*delta/(2*L)**k*minus**(k-i)
        # compute function value
        fnt += beta
        # compute gradient
        gfnt += beta*(k-i)/minus  # no xt yet!
        '''
        # compute temp function value
        FNT[k-i] = delta/(2*L)**k*(minus**(k-i))
        GFNT[k-i] = FNT[k-i]*(k-i)/minus
    # compute function value
    fnt = np.dot(beta_dict[i],FNT)
    # compute gradient
    gfnt = np.dot(beta_dict[i],GFNT) # no xt yet!
    '''
        # compute hessian
        # hfnt += beta*(k-i)*(k-i-1)*(L/2-prod)**(k-i-2)
    return fnt,gfnt,time.time() - run_time


# In[7]:


def w_grad(gfpt,gfnt,yt,at,bt,alphat):
    '''
    Gradient with respect to w
    
    input:
        fpt - positive function at t
        gfpt - positive function gradient at t
        fnt - negative function at t
        gfnt - negative function gradient at t
        yt - sample label at t
        pt - p at t
        at - a at t
        bt - b at t
        alphat - alpha at t
    output:
        gradwt - gradient w.r.t. w at t
    '''
    if yt == 1:
        gradwt = 2*(alphat - at)*gfpt
    else:
        gradwt = 2*(alphat - bt)*gfnt
    return gradwt


# In[8]:


def w_hess(hfpt,hfnt,yt,at,bt,alphat):
    hesswt = 0.0
    if yt == 1:
        hesswt = 2*(alphat - at)*hfpt
    else:
        hesswt = 2*(alphat - bt)*hfnt
    return hesswt


# In[9]:


def proj(wt,R):
    '''
    Projection
    
    input:
        wt - w at t
        R - radius
        
    output:
        proj - projected wt
    '''
    norm = np.linalg.norm(wt)
    if norm > R:
        wt = wt/norm*R
    return wt


# In[10]:


def a_grad(fpt,yt,at):
    '''
    Gradient with respect to a
    
    input:
        fpt - positive function at t
        yt - sample label at t
        pt - p at t
        at - a at t
    
    output:
        gradat - gradient w.r.t a at t
    '''
    gradat = 0.0 
    if yt == 1:
        gradat = 2*(at - fpt)
    else:
        gradat = 2*at
    return gradat


# In[11]:


def b_grad(fnt,yt,bt):
    '''
    Gradient with respect to b
    
    input:
        fnt - negative function at t
        yt - sample label at t
        pt - p at t
        bt - b at t
    
    output:
        gradbt - gradient w.r.t b at t
    '''
    gradbt = 0.0 
    if yt == 1:
        gradbt = 2*bt
    else:
        gradbt = 2*(bt - fnt)
    return gradbt


# In[12]:


def alpha_grad(fpt,fnt,yt,alphat):
    '''
    Gradient with respect to alpha
    '''
    gradalphat = 0.0
    if yt == 1:
        gradalphat = -2*(alphat - fpt)
    else:
        gradalphat = -2*(alphat - fnt)
    return gradalphat



# In[14]:


def split(folder,folders):
    
    if folder > folders:
        print('Exceed maximum folders!')
        return

    # load and split data
    n,d = FEATURES.shape
    # regular portion of each folder
    portion = round(n/folders)
    start = portion*folder
    stop = portion*(folder+1)
    if fabs(stop - n) < portion: # remainder occurs
        train_list = [i for i in range(start)]
        test_list = [i for i in range(start,n)]
        '''
        X_train = FEATURES[:start]
        Y_train = LABELS[:start]
        X_test = FEATURES[start:]
        Y_test = LABELS[start:]
        '''
    else:
        train_list = [i for i in range(start)] + [i for i in range(stop, n)]
        test_list = [i for i in range(start, stop)]
        '''
        mask = [True]*n
        mask[start:stop] = False
        X_train = FEATURES[mask,:]
        Y_train = LABELS[mask]
        X_test = FEATURES[start:stop]
        Y_test = LABELS[start:stop]
    get dimensions of the data
    n,_ = X_train.shape
    # number of epoch
    epoch = T//n+1
    augment by epoch
    X_train_augmented = np.tile(X_train,(epoch,1)) # might have memory burden
    Y_train_augmented = np.tile(Y_train,epoch)
    
    #return X_train_augmented,X_test,Y_train_augmented,Y_test
    #return X_train,X_test,Y_train,Y_test
    '''
    return train_list,test_list

# In[15]:


#def prox(eta,loss,x,y,L,gamma,lam,wj,aj,bj,alphaj,bwt,bat,bbt,balphat):
def prox(eta, loss, index, L, gamma, lam, wj, aj, bj, alphaj, bwt, bat, bbt, balphat):
    '''
    perform proximal guided gradient descent when receive an sample
    '''

    prod = np.dot(wj,FEATURES[index])
    fpt = np.zeros(N+1)
    gfpt = np.zeros(N+1)
    # hfpt = np.zeros(N+1)
    fnt = np.zeros(N+1)
    gfnt = np.zeros(N+1)
    # hfnt = np.zeros(N+1)
    gradwt = 0.0
    gradat = 0.0
    gradbt = 0.0
    gradalphat = 0.0
    # hesswt = 0.0
    run_time = 0.0
    for i in range(N+1):
        fpt[i],gfpt[i] = pos(i,prod,L)
        #print(fpt[i])
        fnt[i],gfnt[i],_ = neg(loss,i,prod,L)
        run_time += _
        gradwt += w_grad(gfpt[i],gfnt[i],LABELS[index],aj[i],bj[i],alphaj[i])# accumulate i
        # hesswt += w_hess(hfpt[i],hfnt[i],y,aj[i],bj[i],alphaj[i])
        gradat = a_grad(fpt[i],LABELS[index],aj[i])
        gradbt = b_grad(fnt[i],LABELS[index],bj[i])
        gradalphat = alpha_grad(fpt[i],fnt[i],LABELS[index],alphaj[i])
        aj[i] = aj[i] - eta*(gradat/(N+1)+gamma*(aj[i]-bat[i]))
        bj[i] = bj[i] - eta*(gradbt/(N+1)+gamma*(bj[i]-bbt[i]))
        alphaj[i] = alphaj[i] + eta*gradalphat/(N+1)
    # hessian = hesswt*np.outer(x,x)
    # eigen,_ = np.linalg.eig(hessian)
    
    # print('minimum eigenvalue: %f' %(np.min(eigen)))
    wj = wj - eta*(gradwt*FEATURES[index]*LABELS[index]/(N+1) + lam*wj + gamma*(wj - bwt))
    wj = proj(wj,L/2)
    #aJ = proj(aJ,1)
    #bJ = proj(bJ,1)
    #alphaJ = proj(alphaJ,1)
    
    return wj,aj,bj,alphaj,run_time


# In[16]:

#def PGSPD(t,loss,X,Y,L,gamma,lam,theta,c,bwt,bat,bbt,balphat):
def PGSPD(t,loss,passing_list,L,gamma,lam,theta,c,bwt,bat,bbt,balphat):
    '''
    Proximally Guided Stochastic Primal Dual Algorithm
    '''
    
    # initialize inner loop variables
    Wt = bwt+0.0
    At = bat+0.0
    Bt = bbt+0.0
    ALPHAt = balphat+0.0
    
    BWt = Wt+0.0
    BAt = At+0.0
    BBt = Bt+0.0
    BALPHAt = ALPHAt+0.0
    
    ETAt = c/(t**theta)

    run_time = 0.0
    # inner loop update at j
    for j in range(t): 
        # update inner loop variables
        Wt, At, Bt, ALPHAt, _ = prox(ETAt, loss, passing_list[j], L, gamma, lam, Wt, At, Bt, ALPHAt, bwt, bat, bbt, balphat)
        #Wt,At,Bt,ALPHAt,_ = prox(ETAt,loss,X[j],Y[j],L,gamma,lam,Wt,At,Bt,ALPHAt,bwt,bat,bbt,balphat)
        run_time += _
        BWt += Wt
        BAt += At
        BBt += Bt
        BALPHAt += ALPHAt
        
    # update outer loop variables
    bwt = BWt/t
    bat = BAt/t
    bbt = BBt/t
    balphat = BALPHAt/t
    
    return bwt,bat,bbt,balphat,run_time


# In[17]:


def SOLAM(t,loss,batch,X,Y,L,lam,theta,c,wt,at,bt,alphat):
    '''
    Stochastic Online AUC Maximization step
    
    input:
        T - total number of iteration
        F - objective function value
        loss - loss function
        pt - p at t
        wt - w at t
        at - a at t
        bt - b at t
        alphat - alpha at t
    output:
        W - record of each wt
        A - record of each at
        B - record of each bt
        ALPHA - record of each alphat
    '''

    # Loop in the batch
    peta = c/(t**theta)
    deta = sqrt(log(T*(T+1)/2/batch)/(T*(T+1)/2/batch))

    for k in range(batch):
        
        # Update wt,at,bt
        prod = np.dot(wt,X[k])
        fpt = np.zeros(N+1)
        gfpt = np.zeros(N+1)
        fnt = np.zeros(N+1)
        gfnt = np.zeros(N+1)
        gradwt = 0.0
        gradat = 0.0
        gradbt = 0.0
        gradalphat = 0.0
        
        for i in range(N+1): # add up info of each i
            fpt[i],gfpt[i] = pos(i,prod,L) # partial info
            fnt[i],gfnt[i] = neg(loss,i,prod,L)
            gradwt += w_grad(gfpt[i],gfnt[i],Y[k],at[i],bt[i],alphat[i])
            gradat = a_grad(fpt[i],Y[k],at[i])
            gradbt = b_grad(fnt[i],Y[k],bt[i])
            gradalphat = alpha_grad(fpt[i],fnt[i],Y[k],alphat[i])
            at[i] -= deta*gradat/(N+1)/batch
            bt[i] -= deta*gradbt/(N+1)/batch
            alphat[i] += deta*gradalphat/(N+1)/batch
        
        wt = wt - peta*(gradwt*Y[k]*X[k]/(N+1)/batch + lam*wt) # step size as 1/t gradient descent
        
    wt = proj(wt,L/2)    
        
    return wt,at,bt,alphat


# In[69]:


#def demo(X_train_augmented,X_test,Y_train_augmented,Y_test,loss,alg,gamma=0.1,lam=1.0,theta=0.1,c = 1.0):
def demo(train_list, test_list, loss, alg, gamma=0.01, lam=10.0, theta=0.25, c=10.0):
    '''
    Run it to get results
    '''
    # define loss function
    if loss == 'hinge':
        L = 2*sqrt(2/lam)
        loss = lambda x: max(0,1+L-2*L*x)
    elif loss == 'logistic':
        L = 2*sqrt(2*log(2)/lam)
        loss = lambda x:log(1+exp(L-2*L*x))
    else:
        print('Wrong loss function!')
        return
    
    # get dimensions of the data
    num = len(train_list)
    _,d = FEATURES.shape
    # initialize outer loop variables
    WT = np.zeros(d) # d is the dimension of the features
    AT = np.zeros(N+1)
    BT = np.zeros(N+1)
    ALPHAT = np.zeros(N+1)

    # record auc
    roc_auc = np.zeros(T)
    # record time elapsed
    start_time = time.time()
    run_time = 0.0

    for t in range(1,T+1):    
        if alg == 'PGSPD':
        
            if t<num:
                begin = (t*(t-1)//2)%num
                end = (t*(t+1)//2)%num
                if begin < end:
                    tr_list = [i for i in range(begin,end)]
                    #x_train = X_train_augmented[begin:end]
                    #y_train = Y_train_augmented[begin:end]
                else: # need to think better
                    tr_list = [i for i in range(begin,num)] + [i for i in range(end)]
                    #x_train = np.append(X_train_augmented[begin:],X_train_augmented[:end],axis=0)
                    #a = aay_train = np.append(Y_train_augmented[begin:],Y_train_augmented[:end],axis=0)
                shuffle(tr_list) # shuffle works in place
                #x_train, y_train = shuffle(x_train,y_train)
                # update outer loop variables
                #WT,AT,BT,ALPHAT,_ = PGSPD(t,loss,x_train,y_train,L,gamma,lam,theta,c,WT,AT,BT,ALPHAT)
                WT, AT, BT, ALPHAT, _ = PGSPD(t, loss, tr_list, L, gamma, lam, theta, c, WT,
                                              AT, BT, ALPHAT)
                run_time += _
            else:
                tr_list = [i for i in range(num)]
                shuffle(tr_list)
                #x_train, y_train = shuffle(X_train_augmented,Y_train_augmented)
                WT,AT,BT,ALPHAT,_ = PGSPD(num,loss,tr_list,L,gamma,lam,theta,c,WT,AT,BT,ALPHAT)
                run_time += _
                
        elif alg == 'SOLAM':

            # sample a point
            begin = (t-1)*batch%num
            end = t*batch%num
            if begin < end:
                x_train = X_train_augmented[begin:end]
                y_train = Y_train_augmented[begin:end]
            else: # need to think better
                x_train = np.append(X_train_augmented[begin:],X_train_augmented[:end],axis=0)
                y_train = np.append(Y_train_augmented[begin:],Y_train_augmented[:end],axis=0)
            WT,AT,BT,ALPHAT = SOLAM(t,loss,batch,x_train,y_train,L,lam,theta,c,WT,AT,BT,ALPHAT)
        
        else:
            print('Wrong algorithm!')
            return
        
        try:
            roc_auc[t-1] = roc_auc_score(LABELS[test_list], np.dot(FEATURES[test_list],WT))
        except RuntimeWarning:
            print('Something is wrong bruh! Look at sum of WT: %f' %(sum(WT)))
            return WT,AT,BT,ALPHAT,roc_auc
        if t%100 == 0:
            elapsed_time = time.time() - start_time
            print('gamma: %.2f lam: %.2f theta: %.2f c: %.2f iteration: %d AUC: %.2f time eplapsed: %.2f/%.2f'
                  %(gamma,lam,theta,c,t,roc_auc[t-1],elapsed_time,run_time))
            start_time = time.time()
            run_time = 0.0
    
            
    return WT,AT,BT,ALPHAT,roc_auc


# In[19]:


def cv(loss,alg,folders=5,gamma=0.01,lam=10.0,theta=0.25,c=10.0):
    '''
    Cross validation
    '''
    # Load data set
    #FEATURES,LABELS = loader(dataset)
    
    # record auc
    AUC_ROC = np.zeros(folders)
    
    # cross validation
    for folder in range(folders):
        print('folder = %d' %(folder))
        #X_train_augmented,X_test,Y_train_augmented,Y_test = split(FEATURES,LABELS,folder,folders)
        training,testing = split(folder, folders)
        
        #_,_,_,_,roc_auc = demo(X_train_augmented,X_test,Y_train_augmented,Y_test,loss,alg,gamma=gamma,lam=lam,theta=theta,c=c)
        _, _, _, _, roc_auc = demo(training, testing, loss, alg, gamma=gamma,
                                   lam=lam, theta=theta, c=c)
        AUC_ROC[folder] = max(roc_auc)
    print('auc score: %f +/- %f' %(np.mean(AUC_ROC),np.std(AUC_ROC)))
    return AUC_ROC


# In[20]:


def single_run(para):
    folder,gamma,lam,theta,c,paras = para
    training,testing,loss,alg = paras
    #X_train_augmented,X_test,Y_train_augmented, Y_test,loss,alg = paras
    _,_,_,_,roc_auc = demo(training,testing,loss,alg,gamma=GAMMA[gamma],lam=LAM[lam],theta=THETA[theta],c=C[c])
    return folder,gamma,lam,theta,c, max(roc_auc)
    
def gs(loss,alg,folders=5,GAMMA=[0.01],LAM=[10.0],THETA=[0.25],C=[10.0]):
    '''
    Grid search! Wuss up fellas?!
    And we are using multiprocessing, fancy!
    '''
    # number of cpu want to use
    num_cpus = 1
    # Load data set
    #FEATURES,LABELS = loader(dataset)
    # record auc
    AUC_ROC = np.zeros((folders,len(GAMMA),len(LAM),len(THETA),len(C)))
    # record parameters
    input_paras = []
    # grid search prepare
    for folder in range(folders):
        training,testing = split(folder,folders)
        paras = training,testing,loss,alg
        for gamma,lam,theta,c in product(range(len(GAMMA)),range(len(LAM)),range(len(THETA)),range(len(C))):
            input_paras.append((folder,gamma,lam,theta,c,paras))
    print('dataset: %s loss: %s algorithm: %s how many paras: %d' % (dataset,loss,alg,len(input_paras)))
    # grid search run on multiprocessors
    with mp.Pool(processes=num_cpus) as pool:
        results_pool = pool.map(single_run,input_paras)
        pool.close()
        pool.join()
        # save results
        for folder,gamma,lam,theta,c, auc_roc in results_pool:
            AUC_ROC[folder,gamma,lam,theta,c] = auc_roc
    '''
    MEAN = np.zeros((len(GAMMA),len(LAM),len(THETA),len(C)))
    STD = np.zeros((len(GAMMA),len(LAM),len(THETA),len(C)))
    for gamma,lam,theta,c in product(range(len(GAMMA)),range(len(LAM)),range(len(THETA)),range(len(C))):
        MEAN[gamma,lam,theta,c] = np.mean(AUC_ROC[:,gamma,lam,theta,c])
        STD[gamma,lam,theta,c] = np.std(AUC_ROC[:,gamma,lam,theta,c])
    '''
    return AUC_ROC


if __name__ == '__main__':

    # Read data from hdf5 file
    dataset = 'diabetes'
    hf = h5py.File('%s.h5' %dataset, 'r')
    FEATURES = hf['FEATURES'][:]
    LABELS = hf['LABELS'][:]
    hf.close()

    # Define hyper parameters
    N = 10
    T = 1000
    batch = 1

    # Define model parameters
    GAMMA = [0.01]
    LAM = [10]
    THETA = [0.25]
    C = [10]

    # Define hyper parameters
    loss = 'hinge'
    alg = 'PGSPD'

    run_time = time.time()
    diabetes_hinge = gs(loss,alg)

    print('total elapsed time: %f' %(time.time() - run_time))

    np.save('diabetes_hinge', diabetes_hinge)


