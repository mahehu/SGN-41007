# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:38:21 2015

@author: hehu
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

if __name__ == "__main__":
    
    plt.close("all")

    N = 100
    n = np.arange(0, 100)
    f = 0.1
    sigma = np.sqrt(0.5)
    
    x0 = np.sin(2 * np.pi * f * n)
    
    x = np.zeros(900)
    x[500:600] = x0
    
    fig,ax = plt.subplots(4, 1)
    plt.tight_layout()
    
    ax[0].plot(x)
    ax[0].set_title("Noiseless Signal")
    
    xn = x + sigma * np.random.randn(x.size)    
    
    ax[1].plot(xn)
    ax[1].set_title("Noisy Signal")
        
    y = np.convolve(xn, x0, 'same')

    ax[2].plot(y)
    ax[2].set_title("Deterministic Detector")
    
    #plt.savefig("../images/sinusoidDetection.pdf", bbox_inches = "tight")

    #plt.savefig("../../Exercises/Ex3/sinusoidDetection.pdf", bbox_inches = "tight")

    # Random signal version
    
    #plt.close("all")

    #fig,ax = plt.subplots(3, 1)
    #plt.tight_layout()
    
    #ax[0].plot(x)
    #ax[0].set_title("Noiseless Signal")
    
    xn = x + sigma * np.random.randn(x.size)    
    
    #ax[1].plot(xn)
    #ax[1].set_title("Noisy Signal")
        
    h = np.exp(-2 * np.pi * 1j * f * n)
    y = np.abs(np.convolve(h, xn, 'same'))

    ax[3].plot(y)
    ax[3].set_title("Stochastic Detector")
    
    #plt.savefig("../images/rayleighSinusoid.pdf", bbox_inches = "tight")
    
    plt.savefig("../../Exercises 2019b/Ex2/sinusoidDetection.pdf", bbox_inches = "tight")

    f2 = plt.figure(figsize = [6,2])
    plt.plot(x, linewidth=1)
    plt.title("Transmitted signal $s[n]$")
    #plt.savefig("../../Exercises/Ex3/sinusoid.pdf", bbox_inches = "tight")    

    #plt.savefig("../images/clean_sinusoid.pdf", bbox_inches = "tight")    

    f3 = plt.figure(figsize = [6,2])
    sigma = np.sqrt(0.2)
    xn = x + sigma * np.random.randn(x.size)    
    plt.plot(xn, linewidth=1)
    plt.title("Received signal $s[n] + w[n]$")
    #plt.savefig("../../Exercises/Ex3/sinusoid.pdf", bbox_inches = "tight")    

    #plt.savefig("../images/noisy_sinusoid.pdf", bbox_inches = "tight")    