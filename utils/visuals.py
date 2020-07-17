import pandas as pd
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt

def tsplot(y, title, lags=None, figsize=(12,8)): 
    """
    ACF and PACF plots as time series EDA 
    """
    fig = plt.figure(figsize=figsize)
    layout = (2,2)
    ts_ax = plt.subplot2grid(layout, (0,0))
    hist_ax = plt.subplot2grid(layout, (0,1))
    acf_ax = plt.subplot2grid(layout, (1,0))
    pacf_ax = plt.subplot2grid(layout, (1,1))
    
    y.plot(ax=ts_ax) 
    ts_ax.set_title(title, fontsize=14, fontweight='bold')
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    plt.tight_layout()
    
    return ts_ax, acf_ax, pacf_ax

def lagged_correlation(pred, target, max_lag=80, min_overlap=50, plot=False):
    """
    Takes two timeseries of the same length and output their lagged correlation 
    """
    assert len(pred) == len(target), 'time series are of different lengths'
    assert len(pred) >= min_overlap, 'time series are shorter than minimum overlap'

    corr = []
    max_lag = min(max_lag, len(pred) - min_overlap + 1)
    for i in range(max_lag): 
        corr.append(np.corrcoef(pred[i: ], target[:len(pred) -i])[0,1]) 
    
    if plot == True: 
        pd.Series(corr).plot(title='Lagged Correlation')
        
    return(corr)

class SliceGenerator(): 
    def __init__(self, model, indata, outdata):
        self.model = model 
        self.indata = indata
        self.prediction = None
        self.outdata = outdata
        self.at = 0 
        self.increment = 100
    
    def now(self):
        r = np.arange(self.at, self.at + self.increment)
        plt.plot(self.outdata[r])
        plt.plot(self.model(self.indata[r]).squeeze().detach().numpy())
        
    def right(self): 
        self.at += self.increment 
        if self.at > self.outdata.shape[0]: 
            print('Index out of range')
            return None   
        self.now()
    
    def left(self): 
        self.at -= self.increment 
        if self.at < 0: 
            print('Index out of range')
            return None   
        self.now()
    
    def everything(self): 
        plt.plot(self.outdata)
        self.prediction = self.model(self.indata).squeeze().detach().numpy()
        plt.plot(self.prediction)        