# -*- coding: utf-8 -*-
"""
Created on Mon Dec 07 13:43:24 2015

@author: hehu
"""


import matplotlib.pyplot as plt
import numpy as np

def gaussian(x, mu, sig):
    multiplier = (1/np.sqrt(2*np.pi*sig**2.))
    return multiplier * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def gaussian_log(x, mu, sig):
    multiplier = (1/np.sqrt(2*np.pi*sig**2.))
    return np.log(multiplier) + (-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
if __name__ == "__main__":
    
    x = np.linspace(-3, 9, 100)
    y1 = gaussian(x, 3, 1)
    
    ax1 = plt.subplot(211)
    p1 = plt.plot(x, y1, linewidth = 2, label = "Likelihood")
    ax1.set_xlabel('A')
    ax1.set_title('PDF of A assuming x[0] = 3')
    
    plt.savefig("../images/PDF_A.pdf", bbox_inches = "tight")
    
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    
    ax2 = ax1.twinx()
    p2 = plt.plot(x, gaussian_log(x, 3, 1), 'r', linewidth = 2, label = "Log-likelihood")
   
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    
    lns = p1 + p2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=1)

    plt.savefig("../images/PDF_A_LL.pdf", bbox_inches = "tight")
    
    ##########################
    # Second picture: log likelihood for a sample of 50 points
    
    np.random.seed(1665)
    
    numSamples = 50
    
    x0 = 5 + np.random.randn(numSamples)
    
    x = np.linspace(5-3, 5+3, 200)
    likelihood = []
    log_likelihood = []
    
    for A in x:
        likelihood.append(gaussian(x0, A, 1).prod())
        log_likelihood.append(gaussian_log(x0, A, 1).sum())
    
    plt.figure()
    
    ax1 = plt.subplot(211)
    p1 = ax1.plot(x, likelihood, linewidth = 2, label = "Likelihood")
    ax1.set_xlabel('A')
    
    maxIdx = x[np.argmax(log_likelihood)]
    
    ax1.set_title('Likelihood of A (max at %.2f)' % maxIdx)
    
    p2 = ax1.plot(x0, 0.5*np.mean(likelihood) * np.ones_like(x0), 'go', label = "Samples")
    
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    
    ax2 = ax1.twinx()
    p3 = plt.plot(x, log_likelihood, 'r', linewidth = 2, label = "Log-likelihood")
   
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    
    lns = p2 + p1 + p3
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=1)
   
    plt.savefig("../images/PDF_A_LL_full.pdf", bbox_inches = "tight")
    
    print(x0.mean())
    