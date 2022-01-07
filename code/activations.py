# -*- coding: utf-8 -*-
"""
Created on Fri Feb 03 10:59:42 2017

@author: hehu
"""

import matplotlib.pyplot as plt
import numpy as np

def relu(x):
    
    return np.maximum(0, x)

def logsig(x):
    
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    
    fig = plt.figure(figsize = [6,4])
    
    x = np.linspace(-5,5,100)
    
    plt.plot([-5,5], [0,0], 'k-', linewidth = 2)
    plt.plot([0,0], [-1,5], 'k-', linewidth = 2)
    
    plt.plot(x, relu(x), label = "ReLU", linewidth = 2)    
    plt.plot(x, np.tanh(x), 'g--', label = "Tanh", linewidth = 2, alpha = 0.4)
    plt.plot(x, logsig(x), 'r--', label = "LogSig", linewidth = 2, alpha = 0.4)
    
    plt.legend(loc = "best")
    plt.axis('equal')
    
    plt.savefig("../images/relu.pdf", bbox_inches = "tight")