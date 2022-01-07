# -*- coding: utf-8 -*-
"""
Created on Mon Dec 07 11:05:34 2015

@author: hehu
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

if __name__ == "__main__":
   
    plt.close("all")
    
    np.random.seed(43)
    numSamples = 200
    
    x = np.random.randn(numSamples)
    plt.figure(1)
    plt.subplot(211)
    
    plt.plot(x, 'og')
    plt.savefig('../images/GaussianRV.pdf', bbox_inches = 'tight')
    
    mean_estimates_1 = []
    mean_estimates_2 = []
    
    for iteration in range(1000):
        
        # Create a random sample of data from N(0,1)
        x = np.random.randn(numSamples)
        
        # Compute the estimate of the mean using the sample mean
        aHat_1 = x.mean() # or np.mean(x)
        
        # Compute the estimate using "first sample estimator"
        aHat_2 = x[0]
        
        # Append the computed values to our lists:
        mean_estimates_1.append(aHat_1)
        mean_estimates_2.append(aHat_2)
        
    # Plot the empirical distributions
    plt.figure()    
    subfig_1 = plt.subplot(211)
    subfig_1.hist(mean_estimates_1, normed = True)
    
    subfig_1.set_title("Distribution of the sample mean estimator")
    subfig_1.axis([-3,3,0,5])

    subfig_1.annotate('Small Variance', 
             xy=(0.25, 0.5), 
             xytext=(1, 2), 
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
            
    ax = plt.subplot(212)
    plt.hist(mean_estimates_2, normed = True)
    plt.title("Distribution of the first sample estimator")
    plt.axis([-3,3,0,5])

    ax.annotate('Large Variance', 
             xy=(0.25, 0.5), 
             xytext=(1, 2), 
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
            
    plt.savefig("../images/2Estimators.pdf", bbox_inches = 'tight')
    