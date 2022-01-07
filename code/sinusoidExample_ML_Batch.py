# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:58:23 2015

@author: hehu
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

if __name__ == "__main__":

    matplotlib.rcdefaults()
    font = {'family' : 'sans-serif'}
    matplotlib.rc('font', **font)
    
    np.random.seed(43)
    
    N = 160
    n = np.arange(N)
    f0 = 0.06752728319488948
    sigmaSq = 1.2
    phi = 0.6090665392794814
    A = 0.6669548209299414
    
    x0 = A * np.cos(2 * np.pi * f0 * n)

    A_estimates = []
    f0_estimates = []
    phi_estimates = []
    numIterations = 1000
    
    for iteration in range(numIterations):
        x = x0 + sigmaSq * np.random.randn(x0.size)
    
        fRange = np.linspace(0, 0.5, 100)
        
        bestScore = 0
        fHat = None
        
        for f in fRange:
            
            expVec = np.exp(-2 * np.pi * 1j * f * n)
            score = np.abs(np.sum(x * expVec))
            
            if score > bestScore:
                fHat = f
                bestScore = score
                        
        expVec = np.exp(-2 * np.pi * 1j * fHat * n)
        aHat = 2.0 / N * np.abs(np.sum(x * expVec))
    
        sinVec = np.sin(2* np.pi * fHat * n)
        cosVec = np.cos(2* np.pi * fHat * n)
        phiHat = np.arctan(-np.sum(x * sinVec) / np.sum(x * cosVec))
        
        A_estimates.append(aHat)
        f0_estimates.append(fHat)
        phi_estimates.append(phiHat)
        
    figsize = [8,11]
    fig, ax = plt.subplots(3, 1, figsize=figsize)

    ax[0].hist(f0_estimates, bins = 50)
    ymin, ymax = ax[0].get_ylim()
    ax[0].plot([f0,f0], [ymin,ymax], 'r--', linewidth=2)
    ax[0].set_title("%d estimates of parameter $f_0$ (true = %.2f; avg = %.2f)" % (numIterations, f0, np.mean(f0_estimates)))
    
    ax[1].hist(A_estimates, bins = 50)
    ymin, ymax = ax[1].get_ylim()
    ax[1].plot([A, A], [ymin,ymax], 'r--', linewidth=2)
    ax[1].set_title("%d estimates of parameter A (true = %.2f; avg = %.2f)" % (numIterations, A, np.mean(A_estimates)))
    
    ax[2].hist(phi_estimates, bins = 50)
    ymin, ymax = ax[2].get_ylim()
    ax[2].plot([phi, phi], [ymin,ymax], 'r--', linewidth=2)
    ax[2].set_title("%d estimates of parameter $\phi$ (true = %.2f; avg = %.2f)" % (numIterations, phi, np.mean(phi_estimates)))
    
    #plt.savefig("../images/ML_sinusoid_batch.pdf", bbox_inches = "tight")
    plt.show()
    