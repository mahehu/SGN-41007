# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:01:16 2015

@author: hehu
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
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
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
def visualize(w, X, y, offset = 0, title = "", show_threshold = False, show_hist = True):
    
    fig, ax = plt.subplots(figsize=[5,4])
    plt.axis('equal')
    
    # create a mesh to plot in

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
    if clf is not None:

        h = .01  # step size in the mesh
        
        xx, yy = np.meshgrid(np.arange(x_min-1, x_max+1, h),
                             np.arange(y_min-1, y_max+1, h))
    
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])            
        # Put the result into a color plot
    
        Z = Z.reshape(xx.shape)
        if show_threshold:
            ax.contourf(xx, yy, Z, cmap='bwr', alpha=0.5) #plt.cm.Paired cmap='bwr',

        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()

        X1 = X[y==1, :]
        X2 = X[y==0, :]
        
        w = clf.coef_.ravel()
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
            
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    X1 = X[y==1, :]
    X2 = X[y==0, :]
    
    ax.plot(X1[:, 0], X1[:, 1], 'ro', zorder = 1, alpha = 0.6)
    ax.plot(X2[:, 0], X2[:, 1], 'bx', zorder = 1)
                
def logistic(x):
    y = 1 / (1 + np.exp(-x))
    return y
    
def show_hist(clf, X, y, logit = False):

    X1 = X[y==0, :]    
    X2 = X[y==1, :]    
    
    w = clf.coef_.T
    b = clf.intercept_

    X1_p = np.dot(X1, w) + b
    X2_p = np.dot(X2, w) + b
    X_p = np.dot(X, w) + b
    
    if logit:
        X1_p = logistic(X1_p)
        X2_p = logistic(X2_p)
        X_p = logistic(X_p)
        
    proj_min = np.min(X_p)
    proj_max = np.max(X_p)

    bins = np.linspace(proj_min, proj_max, 40)
    plt.hist(X1_p, bins = bins, color = 'red', alpha = 0.5)
    plt.hist(X2_p, bins = bins, color = 'blue', alpha = 0.5)    

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
    
    clf = LogisticRegression()
    clf.fit(X, y)
    
    visualize(clf, X, y)
    plt.title("Projection")
    plt.savefig("../images/LR_proj.pdf", transparent = True, bbox_inches = "tight")

    fig = plt.figure(figsize = [5,2])
    show_hist(clf, X, y, logit = False)
    plt.title("Class Score")
    plt.savefig("../images/LR_score.pdf", transparent = True, bbox_inches = "tight")
    
    fig = plt.figure(figsize = [5,2])
    show_hist(clf, X, y, logit = True)
    plt.title("Class Probability")        
    plt.savefig("../images/LR_prob.pdf", transparent = True, bbox_inches = "tight")
    
    plt.figure(figsize=[5,2])
    x = np.linspace(-10,10)
    y = logistic(x)
    y = 0.01 + 0.98 * y
    plt.plot(x, y, linewidth = 3)
    plt.title("Logistic Sigmoid Function")
    plt.savefig("../images/sigmoid.pdf", transparent = True, bbox_inches = "tight")
    
    plt.figure(figsize=[5,2])
    x = np.linspace(-10,10)
    y = np.tanh(x)
    y = 0.01 + 0.98 * y
    plt.plot(x, y, linewidth = 3)
    plt.title("Tanh Sigmoid Function")
    plt.savefig("../images/sigmoid_tanh.pdf", transparent = True, bbox_inches = "tight")

