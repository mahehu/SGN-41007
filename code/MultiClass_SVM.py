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
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from scipy.linalg import eig

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
def visualize(clf, X, y):
    
    fig, ax = plt.subplots(figsize=[5,3])
        
    X1 = X[y==1, :]
    X2 = X[y==2, :]
    X3 = X[y==3, :]
    
    ax.plot(X1[:, 0], X1[:, 1], 'ro', zorder = 1, alpha = 0.6)
    ax.plot(X2[:, 0], X2[:, 1], 'bx', zorder = 1)
    ax.plot(X3[:, 0], X3[:, 1], 'cs', zorder = 1)
    
    # create a mesh to plot in
    
    if clf is not None:

        h = .01  # step size in the mesh
        
        #x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        #y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
        x_min, x_max = 0,5
        y_min, y_max = -6,1
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])            

        # Put the result into a color plot
    
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap="Paired", alpha=0.5) #plt.cm.Paired cmap='bwr',
              
        ax.set_ylim([y_min, y_max])
        ax.set_xlim([x_min, x_max])
        
        X = np.concatenate((X1, X2, X3), axis = 0)
        

        
def generate_data(N):

    X1 = np.random.randn(2,N)
    X2 = np.random.randn(2,N)
    X3 = np.random.randn(2,N)
    
    M1 = np.array([[1.5151, -0.1129], [0.1399, 0.6287]])
    M2 = np.array([[0.8602, 1.2461], [-0.0737, -1.5240]])
    M3 = np.array([[2.202, -.2461], [0.0737, .5240]])
    
    T1 = np.array([-1, 1]).reshape((2,1))
    T2 = np.array([-5, 2]).reshape((2,1))
    T3 = np.array([-3, 4]).reshape((2,1))
    
    X1 = np.dot(M1, X1) + np.tile(T1, [1,N])
    X2 = np.dot(M2, X2) + np.tile(T2, [1,N])
    X3 = np.dot(M3, X3) + np.tile(T3, [1,N])

    X1 = X1[::-1,:]    
    X2 = X2[::-1,:]
    X3 = X3[::-1,:]

    return X1, X2, X3
    
if __name__ == "__main__":
        
    plt.close("all")
    
    # Generate random training data 
        
    N = 200
    
    np.random.seed(2015)
    
    X1, X2, X3 = generate_data(N)
    X1 = X1.T
    X2 = X2.T
    X3 = X3.T
    
    X = np.concatenate((X1, X2, X3))
    y = np.concatenate((np.ones(N), 2*np.ones(N), 3*np.ones(N)))    
    
    visualize(None, X, y)
    plt.savefig("../images/3Class_SVM.pdf", bbox_inches = "tight", transparent = True)
    
    clf_ova = OneVsRestClassifier(LinearSVC())
    clf_ova.fit(X, y)
    
    plt.close("all")    
    
    visualize(clf_ova, X, y)
    
    plt.title("Linear SVM with OvA Wrapper")
    plt.savefig("../images/3Class_SVM_classes_OvA.pdf", bbox_inches = "tight", transparent = True)
    
    clf_ovo = OneVsOneClassifier(LinearSVC())
    clf_ovo.fit(X, y)  
    
    visualize(clf_ovo, X, y)
    plt.title("Linear SVM with OvO Wrapper")
    plt.savefig("../images/3Class_SVM_classes_OvO.pdf", bbox_inches = "tight", transparent = True)
    