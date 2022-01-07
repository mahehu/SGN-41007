# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:58:23 2015

@author: hehu
"""

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

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
    figsize = [6,2]
    
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(n, x0, 'b-', linewidth = 3)
    ax.axis([0,160,-4,4])
    ax.grid(True)
    
    plt.savefig("../images/MLSinusoidOrig.pdf",
                bbox_inches='tight', 
                transparent=True)
    
    x = x0 + sigmaSq * np.random.randn(x0.size)
    
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(n, x, 'go', linewidth = 3)
    ax.axis([0,160,-4,4])
    ax.grid(True)
    plt.savefig("../images/MLSinusoidNoisy.pdf",
                bbox_inches='tight', 
                transparent=True)
           
    fRange = np.linspace(0, 0.5, 1000)
    
    scores = []
    frequencies = []
    
    for f in np.linspace(0, 0.5, 1000):
    		
    		expVec = np.exp(-2 * np.pi * 1j * f * n)
    		score = np.abs(np.sum(x * expVec))
    		scores.append(score)
    		frequencies.append(f)
    
    fHat = frequencies[np.argmax(scores)]

    expVec = np.exp(-2 * np.pi * 1j * fHat * n)
    aHat = 2.0 / N * np.abs(np.sum(x * expVec))

    sinVec = np.sin(2* np.pi * fHat * n)
    cosVec = np.cos(2* np.pi * fHat * n)
    phiHat = np.arctan(-np.sum(x * sinVec) / np.sum(x * cosVec))
    
    xHat = aHat * np.cos(2 * np.pi * fHat * n + phiHat)
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(n, x, 'go', alpha = 0.5)
    ax.plot(n, x0, 'b-', linewidth = 3)
    ax.plot(n, xHat, 'r-', linewidth = 3, alpha = 0.6)
    
    title = "$\hat{f}_0$ = %.3f (%.3f); $\hat{A}$ = %.3f (%.3f); $\hat{\phi}$ = %.3f (%.3f)" % (fHat, f0, aHat, A, phiHat, phi)
    plt.title(title)

    #plt.show()
    ax.axis([0,160,-4,4])    
    ax.grid(True)
    plt.savefig("../images/MLSinusoid.pdf",
                bbox_inches='tight', 
                transparent=True)
    
    #########################################
        
    fig, ax = plt.subplots(figsize=figsize)
            
    x = x0 + sigmaSq * np.random.randn(x0.size)
    
    fRange = np.linspace(0, 0.5, 1000)
    
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
    
    xHat = aHat * np.cos(2 * np.pi * fHat * n + phiHat)
    
    ax.plot(n, x, 'go', alpha = 0.5)
    ax.plot(n, x0, 'b-', linewidth = 3)
    ax.plot(n, xHat, 'r-', linewidth = 3, alpha = 0.6)
    title = "$\hat{f}_0$ = %.3f; $\hat{A}$ = %.3f; $\hat{\phi}$ = %.3f" % (fHat, aHat, phiHat)
    ax.set_title(title)
    ax.axis([0,160,-4,4])
    ax.grid(True)
    
    plt.savefig("../images/MLSinusoid1.pdf",
                bbox_inches='tight', 
                transparent=True)
    #########################################
                
    fig, ax = plt.subplots(figsize=figsize)
    x = x0 + sigmaSq * np.random.randn(x0.size)
    
    fRange = np.linspace(0, 0.5, 1000)
    
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
    
    xHat = aHat * np.cos(2 * np.pi * fHat * n + phiHat)
    
    ax.plot(n, x, 'go', alpha = 0.5)
    ax.plot(n, x0, 'b-', linewidth = 3)
    ax.plot(n, xHat, 'r-', linewidth = 3, alpha = 0.6)
    title = "$\hat{f}_0$ = %.3f; $\hat{A}$ = %.3f; $\hat{\phi}$ = %.3f" % (fHat, aHat, phiHat)
    ax.set_title(title)
    ax.axis([0,160,-4,4])
    ax.grid(True)
    
    plt.savefig("../images/MLSinusoid2.pdf",
                bbox_inches='tight', 
                transparent=True)
        #########################################

    fig, ax = plt.subplots(figsize=figsize)
    x = x0 + sigmaSq * np.random.randn(x0.size)
    
    fRange = np.linspace(0, 0.5, 1000)
    
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
    
    xHat = aHat * np.cos(2 * np.pi * fHat * n + phiHat)
    
    ax.plot(n, x, 'go', alpha = 0.5)
    ax.plot(n, x0, 'b-', linewidth = 3)
    ax.plot(n, xHat, 'r-', linewidth = 3, alpha = 0.6)
    title = "$\hat{f}_0$ = %.3f; $\hat{A}$ = %.3f; $\hat{\phi}$ = %.3f" % (fHat, aHat, phiHat)
    ax.set_title(title)
    ax.axis([0,160,-4,4])
    ax.grid(True)
    
    plt.savefig("../images/MLSinusoid3.pdf",
                bbox_inches='tight', 
                transparent=True)
        #########################################

    fig, ax = plt.subplots(figsize=figsize)
    x = x0 + sigmaSq * np.random.randn(x0.size)
    
    fRange = np.linspace(0, 0.5, 1000)
    
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
    
    xHat = aHat * np.cos(2 * np.pi * fHat * n + phiHat)
    
    ax.plot(n, x, 'go', alpha = 0.5)
    ax.plot(n, x0, 'b-', linewidth = 3)
    ax.plot(n, xHat, 'r-', linewidth = 3, alpha = 0.6)
    title = "$\hat{f}_0$ = %.3f; $\hat{A}$ = %.3f; $\hat{\phi}$ = %.3f" % (fHat, aHat, phiHat)
    ax.set_title(title)
    ax.axis([0,160,-4,4])
    ax.grid(True)
    
    plt.savefig("../images/MLSinusoid4.pdf",
                bbox_inches='tight', 
                transparent=True)