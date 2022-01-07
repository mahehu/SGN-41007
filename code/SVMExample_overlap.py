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
import matplotlib

from scipy.linalg import eig

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
def visualize(X, y, clf):
    
    fig, ax = plt.subplots(figsize=[5,6])

    # create a mesh to plot in
    
    if clf is not None:

        h = .01  # step size in the mesh
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
    
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])            
        # Put the result into a color plot
    
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap='bwr', alpha=0.5) #plt.cm.Paired cmap='bwr',

        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
    
        # get the separating hyperplane
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-10, 10, 500)
        yy = a * xx - (clf.intercept_[0]) / w[1]
    
        # plot the parallels to the separating hyperplane that pass through the
        # support vectors
        margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
        yy_down = yy + a * margin
        yy_up = yy - a * margin

        ax.plot(xx, yy, 'k-')
        ax.plot(xx, yy_down, 'k--')
        ax.plot(xx, yy_up, 'k--')
    
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
        
    matplotlib.rcdefaults()
    font = {'family' : 'sans-serif'}
    matplotlib.rc('font', **font)
    
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

    clf = SVC(kernel = 'linear', C = 1)
    clf.fit(X, y)
    
    visualize(X, y, None)
    plt.savefig("../images/SVM_data_overlap.pdf", bbox_inches = "tight", transparent = True)
    
    visualize(X, y, clf)
    plt.savefig("../images/SVM_boundary_overlap.pdf", bbox_inches = "tight", transparent = True)    
    
    for Clog in [-4,-1,5,8]:
        C = 10.0 ** Clog
        clf = SVC(kernel = 'linear', C = C)
        clf.fit(X, y)
        
        visualize(X, y, clf)
        plt.title("SVM with C = $10^{%d}$" % Clog)
        plt.savefig("../images/SVM_C_%d.pdf" % Clog, bbox_inches = "tight", transparent = True)
    
    x = np.linspace(-5,5)
    y = 0.05 + x * (x > 0)
    plt.subplot(311)
    plt.plot(x,y, linewidth=3, label = 'Hinge Loss')
    plt.axis([-5,5,0,5])
    plt.legend(loc = 'best')
    plt.xlabel("$\longleftarrow$ Right Side         Wrong Side $\longrightarrow$")
    plt.savefig("../images/hinge_loss.pdf", bbox_inches = "tight", transparent = True)    
    