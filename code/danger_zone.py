# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 12:01:45 2017

@author: hehu
"""

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
        
    fig,ax = plt.subplots(1,1, figsize = [10,10])
    ax.axis([0,10,0,10])
    c = plt.Circle(xy = (3,7), radius = 4, alpha = 0.4, fc = 'blue')
    ax.add_patch(c)
    
    fontsize = 15
    plt.text(1.5, 7.5, 'Hacker\nSkills', ha = 'center', va = 'center', fontsize = fontsize)
    
    c = plt.Circle(xy = (7,7), radius = 4, alpha = 0.4, fc = 'red')
    ax.add_patch(c)
    plt.text(9, 7.5, 'Substance', ha = 'center', va = 'center', fontsize = fontsize)
    
    y = 7 - 2*np.sqrt(3)
    c = plt.Circle(xy = (5, y), radius = 4, alpha = 0.4, fc = 'green')
    ax.add_patch(c)
    plt.text(5, 1.5, 'Math &\n Statistics', ha = 'center', va = 'center', fontsize = fontsize)
    
    ax.annotate("Danger Zone", 
                 xy = (5, 9), 
                 xytext=(6, 11.5), 
                 size=fontsize,
                 bbox=dict(boxstyle="round4", fc="w", ec = "g"),
                 arrowprops=dict(arrowstyle="simple",
                                 connectionstyle="arc3,rad=0.2",
                                 shrinkA = 0,
                                 shrinkB = 8,
                                 fc = "g",
                                 ec = "g"),
                 horizontalalignment='center', 
                 verticalalignment='middle')
    
    ax.annotate("Model Based Research\n(Biology, Physics,...)", 
                 xy = (8, 4), 
                 xytext=(11, 0), 
                 size=fontsize,
                 bbox=dict(boxstyle="round4", fc="w", ec = "g"),
                 arrowprops=dict(arrowstyle="simple",
                                 connectionstyle="arc3,rad=0.2",
                                 shrinkA = 0,
                                 shrinkB = 8,
                                 fc = "g",
                                 ec = "g"),
                 horizontalalignment='center', 
                 verticalalignment='middle')
    
    ax.annotate("Machine Learning", 
                 xy = (2, 4), 
                 xytext=(-1, 0), 
                 size=fontsize,
                 bbox=dict(boxstyle="round4", fc="w", ec = "g"),
                 arrowprops=dict(arrowstyle="simple",
                                 connectionstyle="arc3,rad=-0.2",
                                 shrinkA = 0,
                                 shrinkB = 8,
                                 fc = "g",
                                 ec = "g"),
                 horizontalalignment='center', 
                 verticalalignment='middle')
    
    plt.text(5, 6, 'Superman', ha = 'center', va = 'center', fontsize = fontsize)
    
    ax.axis('equal')
    ax.axis('off')
    
    plt.savefig("../images/danger_zone.pdf", bbox_inches = 'tight')
    plt.show()
    