import numpy as np

def create_predwin(a, winsize, pred_target, pred_lag, offset=0): 
    """
    a(np.array): time x dimensions 
    """
    padding = winsize + pred_lag - 1 
    until = a.shape[0] - padding + offset

    y = a[padding - offset:, pred_target]
    X = np.zeros((until, winsize, a.shape[1]))

    for i in np.arange(until): 
        X[i,:,:] = a[i:i+winsize,:]

    # when target offsets predictor
    if offset != 0: 
        X = np.delete(X,pred_target,axis=2)

    return X,y
