import numpy as np

def create_predwin(a, winsize, pred_target, pred_lag): 
    until = a.shape[0] - winsize - pred_lag + 1
    X = np.zeros((until, winsize, a.shape[1]))
    y = a[winsize + pred_lag:, pred_target]
    for i in np.arange(until): 
        X[i,:,:] = a[i:i+winsize,:]
    return X,y