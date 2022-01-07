# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:48:23 2016

@author: hehu
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from cross_validation import get_toy_data
import numpy as np
import matplotlib

def visualize(X, y):
    
    fig, ax = plt.subplots(figsize=[6,6])
        
    # create a mesh to plot in

    h = .05  # step size in the mesh
    
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = log_loss(np.c_[xx.ravel(), yy.ravel()], X, y)
    minidx = np.argmin(Z)
    x0 = xx.ravel()[minidx]
    y0 = yy.ravel()[minidx]
    
    Z = np.reshape(Z, xx.shape)
    
    levels = range(0, 4000, 100)
    CS = plt.contourf(xx, yy, Z, cmap = cm.get_cmap("cool"), levels = levels, alpha=0.3)
    CS = plt.contour(CS, alpha = 0.4)
    
    return (x0,y0)
        
def log_loss(w, X, y):

    loss = []
    
    for row in w:
        loss.append(np.sum(np.log(1 + np.exp(y * np.dot(X, row)))))
        
    return np.array(loss)


def squared_loss(w, X, y):

    loss = []
    
    for row in w:
        loss.append(np.sum((y - np.dot(X, row))**2))
        
    return np.array(loss)
        
if __name__ == "__main__":
    
    matplotlib.rcdefaults()
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    
    np.random.seed(15)
    X, y = get_toy_data(200)
    X -= np.tile(X.mean(axis = 0), [X.shape[0], 1])
    X *= 1.3
    X[:,0] /= 1.3
    y[y==0] = -1
    
    x0, y0 = visualize(X, y)
    
    wmax = 1.0
    color = 'r'
    plt.plot([wmax, 0],[0, wmax], color, linewidth = 3)
    plt.plot([wmax, 0],[0, -wmax], color, linewidth = 3)
    plt.plot([-wmax, 0],[0, wmax], color, linewidth = 3)
    plt.plot([-wmax, 0],[0, -wmax], color, linewidth = 3)
    
    r = 0.9
    angles = np.arange(360)
    plt.plot(r * np.cos(np.pi*angles/180), r * np.sin(np.pi*angles/180), 'b', linewidth = 3)
    
    plt.axis('tight')
    
    xmin, xmax = plt.gca().get_xlim()
    ymin, ymax = plt.gca().get_ylim()
    
    plt.plot([0, 0],[ymin, ymax], 'k--')
    plt.plot([xmin, xmax], [0, 0], 'k--')
    
    plt.annotate("Unregularized\nSolution", 
                 (x0, y0), 
                 xytext=(1, -1.8), 
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
                 
    plt.annotate("L$_1$-Regularized\nSolution", 
                 (0, -1.), 
                 xytext=(1, -1.3), 
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


    plt.annotate("L$_2$-Regularized\nSolution", 
                 (-0.22, -.85), 
                 xytext=(-1.2, -1.3), 
                 size=13,
                 bbox=dict(boxstyle="round4", fc="w", ec = "g"),
                 arrowprops=dict(arrowstyle="simple",
                                 connectionstyle="arc3,rad=0.2",
                                 shrinkA = 0,
                                 shrinkB = 8,
                                 fc = "g",
                                 ec = "g"),
                 horizontalalignment='center', 
                 verticalalignment='middle')                                  
    
    plt.title(r"Logistic Loss Surface for \textbf{w} = (w$_1$, w$_2$)")
    plt.xlabel(r"w$_1$")
    plt.ylabel(r"w$_2$")
    plt.savefig("../images/L1-reg.pdf", bbox_inches = "tight")                 