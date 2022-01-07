# -*- coding: utf-8 -*-
"""
Created on Mon Dec 07 13:43:24 2015

@author: hehu
"""

import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

def gaussian(x, mu, sig):
    multiplier = (1/np.sqrt(2*np.pi*sig**2.))
    return multiplier * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
if __name__ == "__main__":
    
    rc('font',**{'family':'sans-serif'}) #,'sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    x = np.linspace(-3, 9, 100)
    y = gaussian(x, 1, 1)
    
    ax = plt.subplot(211)
    plt.plot(x, y, linewidth = 2, label = "Gaussian with $\mu=1$")

    y = gaussian(x, 0, 1)
    plt.plot(x, y, linewidth = 2, label = "Gaussian with $\mu=0$")
    plt.legend(loc = "best")
    
    plt.plot([2], [0], 'go', clip_on=False)
    
    ax.annotate("Where did this come from?", 
                 [2,0], 
                 xytext=[6,0.15], 
                 size=13,
                 bbox=dict(boxstyle="round4", fc="w", ec = "g"),
                 arrowprops=dict(arrowstyle="simple",
                                 connectionstyle="arc3,rad=-0.2",
                                 shrinkA = 0,
                                 shrinkB = 8,
                                 fc = "g",
                                 ec = "g"),
                 horizontalalignment='center', 
                 verticalalignment='center')

    ax.set_ylim([0,0.5])

    plt.savefig("../images/two_gaussians.pdf", bbox_inches = "tight")
    
