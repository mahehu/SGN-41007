# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:01:16 2015

@author: hehu
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from scipy.linalg import eig

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / np.sqrt(2*np.pi*sig**2)
    
def visualize(w, X, y, offset = 0, title = "", show_threshold = False, show_hist = True):
    
    fig, ax = plt.subplots(figsize=[5,6])
        
    # create a mesh to plot in
    
    if clf is not None:

        h = .01  # step size in the mesh
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
    
       
    X1 = X[y==1, :]
    X2 = X[y==0, :]
    
    ax.plot(X1[:, 0], X1[:, 1], 'ro', zorder = 1, alpha = 0.6)
    ax.plot(X2[:, 0], X2[:, 1], 'bx', zorder = 1)
    
    w = w / np.linalg.norm(w)
    dx, dy = w

    yMin = -100
    yMax = 100
    xMin = -100
    xMax = 100
    
    x0, y0 = np.mean(X1, axis = 0)
    
    plt.plot([xMin, xMax], y0 + (dy / dx) * (np.array([xMin, xMax]) - x0), 'g-', linewidth = 2, zorder = 3)
    
    # In ax + by + c = 0 format:
    a = -(dy / dx)
    b = 1
    c = (dy / dx) * x0 - y0
    
    m1 = np.mean(X1, axis = 0)

    order = np.argsort(np.dot(X, w))
    
    prev_proj = None
    
    for k in range(X.shape[0]):
        idx = order[k]
        
        p = X[idx, :] - m1
        proj = np.dot(w, p)
                
        x1 = X[idx, 0]
        y1 = X[idx, 1]
        x2 = (b*(b*x1 - a*y1) - a*c) / (a**2 + b**2)
        y2 = (a*(-b*x1 + a*y1) - b*c) / (a**2 + b**2)

        length = np.hypot([x1-x2],[y1-y2])

        if (prev_proj is None or proj - prev_proj > 0.05) or length > 0.5:
            ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.2, head_length=0.3, fc='k', ec='k', length_includes_head = True, zorder = 2, alpha = 0.2)
            #ax.plot([x1,x2], [y1,y2])

        prev_proj = proj
                
    plt.axis("equal")

    if show_threshold:
        x1, y1 = np.mean(X1, axis = 0)
        x2, y2 = np.mean(X2, axis = 0)
        
        m1_x = (b*(b*x1 - a*y1) - a*c) / (a**2 + b**2)
        m1_y = (a*(-b*x1 + a*y1) - b*c) / (a**2 + b**2)

        m2_x = (b*(b*x2 - a*y2) - a*c) / (a**2 + b**2)
        m2_y = (a*(-b*x2 + a*y2) - b*c) / (a**2 + b**2)

        T_x = np.mean([m1_x, m2_x])
        T_y = np.mean([m1_y, m2_y])
        
        plt.plot([T_x], [T_y], 'go', zorder = 4, markersize = 10)
        
        plt.annotate('Threshold', 
             xy=(T_x, T_y), 
             xytext=(-1.5, 1), 
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
    
    yMin = -10
    yMax = 2
    xMin = -2
    xMax = 6

    ax.set_ylim([yMin, yMax])
    ax.set_xlim([xMin, xMax])

    if show_hist:
    
        proj_min = np.min(np.dot(X, w))
        proj_max = np.max(np.dot(X, w))
        bins = np.linspace(proj_min, proj_max, 40)
    
        ax2 = plt.axes([0.12, -0.2, 0.78, 0.2])
        ax2.hist(np.dot(X1, w), bins = bins, color = 'red', alpha = 0.5)
    
        ax2.hist(np.dot(X2, w), bins = bins, color = 'blue', alpha = 0.5)    
        ax2.xaxis.set_ticks([])
     
        ax2.set_title(title)     
    
def generate_data(N):

    X1 = np.random.randn(2,N)
    X2 = np.random.randn(2,N)
    
    M1 = np.array([[1.5151, -0.1129], [0.1399, 0.6287]])
    M2 = np.array([[0.8602, 1.2461], [-0.0737, -1.5240]])
    
    T1 = np.array([-1, 1]).reshape((2,1))
    T2 = np.array([-5, 2]).reshape((2,1))
    
    X1 = np.dot(M1, X1) + np.tile(T1, [1,N])
    X2 = np.dot(M2, X2) + np.tile(T2, [1,N])

    X1 = X1[::-1,:]    
    X2 = X2[::-1,:]

    return X1, X2
    
if __name__ == "__main__":
        
    plt.close("all")
    plt.style.use("classic")
    
    # Generate random training data 
        
    N = 200
    
    np.random.seed(2015)
    
    X1, X2 = generate_data(N)
    
    X = np.concatenate((X1.T, X2.T))
    y = np.concatenate((np.ones(N), np.zeros(N)))    
    
    # Generate test sample
    
    np.random.seed(2016)
    
    X1_test, X2_test = generate_data(N)
    
    X_test = np.concatenate((X1_test.T, X2_test.T))
    y_test = np.concatenate((np.ones(N), np.zeros(N)))    

    m1 = np.mean(X1.T, axis = 0)
    m2 = np.mean(X2.T, axis = 0)
    C1 = np.cov(X1.T - np.tile(m1, (X1.shape[1], 1)), rowvar = False)
    C2 = np.cov(X2.T - np.tile(m2, (X2.shape[1], 1)), rowvar = False)
    
    w = np.dot(np.linalg.inv(C1 + C2), (m1 - m2))
    
    SB = np.multiply.outer(m2-m1, m2-m1)
    SW = C2 + C1

    D, V = eig(SB, SW)
    print(V[:, 1])
    print(w)
    
    w = V[:,1]
    T = np.mean([np.dot(m1, w), np.dot(m2, w)])
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    
    visualize(w, X, y, offset = -10, title = "Good Projection: Good Separation")
    plt.savefig("../images/LDA_proj1.pdf", transparent = True, bbox_inches = "tight")

    visualize(w, X, y, offset = -10, title = "", show_threshold = True, show_hist = False)
    plt.savefig("../images/LDA_proj1_threshold.pdf", transparent = True, bbox_inches = "tight")
    
    w[1] = 0.05
    visualize(w, X, y, offset = -10, title = "Poor Projection: Poor separation")
    plt.savefig("../images/LDA_proj2.pdf", transparent = True, bbox_inches = "tight")
    
    