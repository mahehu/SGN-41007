# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:20:20 2015

@author: hehu
"""

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    
    np.random.seed(2015)
    
    x = np.linspace(-10,10,100)
    y = 1 / (1 + np.exp(-x))
    
    x = x + 0.1 * np.random.randn(x.size)
    y = y + 0.1 * np.random.randn(y.size)

    fig, ax = plt.subplots(figsize=[5,6])
    
    ax.plot(x, y, 'ro')
    
    ax.plot([13, 13], [-0.4, 1.4], '--')
    
    plt.xlabel("x")
    plt.ylabel("y")
    
    ax.annotate("Which y coordinate", 
                 [13, 1.0], 
                 xytext=(0.75, 0.35), 
                 textcoords='axes fraction',
                 size=13,
                 bbox=dict(boxstyle="round4", fc="w", ec = "g"),
                 arrowprops=dict(arrowstyle="simple",
                                 connectionstyle="arc3,rad=-0.2",
                                 shrinkA = 0,
                                 shrinkB = 8,
                                 fc = "g",
                                 ec = "g"),
                 horizontalalignment='center', 
                 verticalalignment='middle')
                 
    plt.savefig("../images/regressionExample.pdf")