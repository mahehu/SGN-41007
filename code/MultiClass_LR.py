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

from statsmodels.api import MNLogit

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from scipy.linalg import eig

from sklearn.cross_validation import train_test_split
 

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
def visualize(clf, X, y, Xm = None, predict_proba = True, stFormat = False):
    
    fig, ax = plt.subplots(figsize=[3,3.6])
        
    X1 = X[y==1, :]
    X2 = X[y==2, :]
    X3 = X[y==3, :]
    
    ax.plot(X1[:, 0], X1[:, 1], 'ro', zorder = 1, alpha = 0.6)
    ax.plot(X2[:, 0], X2[:, 1], 'bx', zorder = 1)
    ax.plot(X3[:, 0], X3[:, 1], 's', zorder = 1, color = '#00dd00')
    
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
 
    if clf is not None:

        h = .01  # step size in the mesh
             
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        if predict_proba:
            if stFormat:
                Z = clf.predict(np.c_[xx.ravel() - Xm[0], yy.ravel() - Xm[1]])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
                
            Z = Z.reshape((xx.shape[0], xx.shape[1], 3))
            Z_new = np.zeros_like(Z)
            Z_new[...,0] = Z[...,0]
            Z_new[...,1] = Z[...,2]
            Z_new[...,2] = Z[...,1]

            plt.imshow(Z_new[::-1,...], extent=[x_min, x_max, y_min, y_max], alpha = 0.75)
            
#            colormaps = ['Blues', 'Greens', 'Reds']
#            
#            for channel in range(Z.shape[-1]):
#                cmap = plt.get_cmap(colormaps[channel])
#                ax.contourf(xx, yy, Z[...,channel], 100, cmap=cmap, alpha=0.5)
            
        else:
            if stFormat:
                Z = clf.predict(np.c_[xx.ravel() - Xm[0], yy.ravel() - Xm[1]])
                Z = np.argmax(Z, axis = 1)
            else:
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap="Paired", alpha=0.5) #plt.cm.Paired cmap='bwr',

              
    ax.set_ylim([y_min, y_max])
    ax.set_xlim([x_min, x_max])
    
        
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
    plt.title("Data")
    plt.savefig("../images/LR_3classes_data.pdf", bbox_inches = "tight", transparent = True)
   
    clf = LogisticRegression(C = 1e8)
    clf.fit(X, y)
    
    visualize(clf, X, y, predict_proba = False)
    plt.title("Predicted Classes")
    plt.savefig("../images/LR_3classes.pdf", bbox_inches = "tight", transparent = True)
    
    visualize(clf, X, y, predict_proba = True)
    plt.title("Class Probabilities")
    plt.savefig("../images/LR_3classes_prob.pdf", bbox_inches = "tight", transparent = True)    

    Xm = np.mean(X, axis = 0)
    clf = MNLogit(y, X - np.tile(Xm, (X.shape[0], 1)))
    clf = clf.fit()    
    
    visualize(clf, X, y, Xm, predict_proba = False, stFormat = True)
    plt.title("Predicted Classes")
    plt.savefig("../images/LR_3classes_stf.pdf", bbox_inches = "tight", transparent = True)
    
    visualize(clf, X, y, Xm, predict_proba = True, stFormat = True)
    plt.title("Class Probabilities")
    plt.savefig("../images/LR_3classes_prob_stf.pdf", bbox_inches = "tight", transparent = True)    
    
    # Example of penalty
    
    # Split data to training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Test C values -4, -3, ..., 1
    C_range = 10.0 ** np.arange(-4, 3)
    clf = LogisticRegression()
    
    for C in C_range:
        clf.C = C
        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)
        accuracy = 100.0 * np.mean(y_hat == y_test)
        print "Accuracy for C = %.2e is %.1f %% (||w|| = %.4f)" % \
            (C, accuracy, np.linalg.norm(clf.coef_))
        
        