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
    
def visualize(X, y, clf):
    
    fig, ax = plt.subplots(figsize=[6,6])
    plt.axis('equal')
    
    # create a mesh to plot in

    #x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_min, x_max = -9, 3
    y_min, y_max = -7, 5
        
    if clf is not None:

        h = .01  # step size in the mesh
        
        xx, yy = np.meshgrid(np.arange(x_min-1, x_max+1, h),
                             np.arange(y_min-1, y_max+1, h))
    
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])            
        # Put the result into a color plot
    
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap='bwr', alpha=0.5) #plt.cm.Paired cmap='bwr',

        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
    
        if clf.kernel == "linear":

            # get the separating hyperplane
            w = clf.coef_[0]
            a = -w[0] / w[1]
            xx = np.linspace(-10, 5, 500)
            yy = a * xx - (clf.intercept_[0]) / w[1]
        
            # plot the parallels to the separating hyperplane that pass through the
            # support vectors
            margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
            yy_down = yy + a * margin
            yy_up = yy - a * margin
    
            ax.plot(xx, yy, 'k-')
            ax.plot(xx, yy_down, 'k--')
            ax.plot(xx, yy_up, 'k--')
            
            for svIdx in range(clf.support_vectors_.shape[0]):
                sv = [clf.support_vectors_[svIdx, 0], clf.support_vectors_[svIdx, 1]]
                ax.annotate("Support Vectors", 
                         sv, 
                         xytext=(-6, 3), 
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
    
            # Plot margin
            
            x0 = -0.5
            y0 = a * x0 - (clf.intercept_[0]) / w[1]
            
            distances = np.hypot(x0 - xx, y0 - yy_down)
            minIdx = np.argmin(distances)
            x1 = xx[minIdx]
            y1 = yy_down[minIdx]
            ax.annotate("",
                xy=(x0, y0), xycoords='data',
                xytext=(x1, y1), textcoords='data',
                arrowprops=dict(arrowstyle="<->",
                                connectionstyle="arc3"),
                )
    
            distances = np.hypot(x0 - xx, y0 - yy_up)
            minIdx = np.argmin(distances)
            x2 = xx[minIdx]
            y2 = yy_up[minIdx]
            ax.annotate("",
                xy=(x0, y0), xycoords='data',
                xytext=(x2, y2), textcoords='data',
                arrowprops=dict(arrowstyle="<->",
                                connectionstyle="arc3"),
                )
            
            ax.annotate("Margin", 
                         (0.5*(x0+x1), 0.5*(y0+y1)),
                         xytext=(1.5, -6.7), 
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
    
            ax.annotate("Margin", 
                         (0.5*(x0+x2), 0.5*(y0+y2)),
                         xytext=(1.5, -6.7), 
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
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                    facecolors='none', zorder=10)
        #ax.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

    
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    X1 = X[y==1, :]
    X2 = X[y==0, :]
    
    ax.plot(X1[:, 0], X1[:, 1], 'ro', zorder = 1, alpha = 0.6)
    ax.plot(X2[:, 0], X2[:, 1], 'bx', zorder = 1)
 
def generate_data(N):

    X1 = np.random.randn(2,N)
    X2 = np.random.randn(2,N)
    
    M1 = 0.7*np.array([[1.5151, -0.1129], [0.1399, 0.6287]])
    M2 = 0.7*np.array([[0.8602, 1.2461], [-0.0737, -1.5240]])
    
    T1 = np.array([-1, 1]).reshape((2,1))
    T2 = np.array([-2, -5]).reshape((2,1))
    
    X1 = np.dot(M1, X1) + np.tile(T1, [1,N])
    X2 = np.dot(M2, X2) + np.tile(T2, [1,N])

    X1 = X1[::-1,:]    
    X2 = X2[::-1,:]

    return X1, X2
    
if __name__ == "__main__":
        
    plt.close("all")
    
    # Generate random training data 
        
    N = 200
    
    np.random.seed(2014)
    
    X1, X2 = generate_data(N)
    
    X = np.concatenate((X1.T, X2.T))
    y = np.concatenate((np.ones(N), np.zeros(N)))    
    
    # Generate test sample
    
    np.random.seed(2016)
    
    X1_test, X2_test = generate_data(N)
    
    X_test = np.concatenate((X1_test.T, X2_test.T))
    y_test = np.concatenate((np.ones(N), np.zeros(N)))    

    clf = SVC(kernel = 'linear', C = 100)
    clf.fit(X, y)
    
    visualize(X, y, None)
    plt.savefig("../images/SVM_data.pdf", bbox_inches = "tight", transparent = True)
    
    visualize(X, y, clf)
    plt.savefig("../images/SVM_boundary.pdf", bbox_inches = "tight", transparent = True)    
    
    clf = SVC(kernel = 'poly', degree = 2, C = 1)
    clf.fit(X, y)
    
    visualize(X, y, clf)
    plt.title("SVM with 2nd order Polynomial Kernel")
    plt.savefig("../images/SVM_boundary_poly2.pdf", bbox_inches = "tight", transparent = True)    

    clf = SVC(kernel = 'rbf', C = 1)
    clf.fit(X, y)
    
    visualize(X, y, clf)
    plt.title("SVM with the RBF Kernel")
    plt.savefig("../images/SVM_boundary_RBF.pdf", bbox_inches = "tight", transparent = True)        
