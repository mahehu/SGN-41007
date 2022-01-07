# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:22:08 2017

@author: hehu
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    
    sigma = 0.2
    
    X1 = sigma*np.random.randn(200, 2) + np.tile([[1.4,1.4]], [200,1])
    X2a = sigma*np.random.randn(100, 2) + np.tile([[0.5,2]], [100,1])
    X2b = sigma*np.random.randn(100, 2) + np.tile([[2,0.5]], [100,1])
    
    X = np.vstack((X1, X2a, X2b))
    y = np.hstack((np.zeros(200), np.ones(200)))
    
    plt.plot(X[y==0, 0], X[y==0, 1], 'ro')
    plt.plot(X[y==1, 0], X[y==1, 1], 'bx')
    plt.show()
    
    clf = LDA()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy: %.2f" % (accuracy_score(y_test, y_pred)))
    
    # Attempt to increase dimensionality
    new_var = np.atleast_2d(2*X[:, 0] * X[:, 1]).T
    
    X = np.hstack((X, new_var))
    clf = LDA()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy (3D): %.2f" % (accuracy_score(y_test, y_pred)))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[y==0, 0], X[y==0, 1], X[y==0, 2], 'ro')
    ax.plot(X[y==1, 0], X[y==1, 1], X[y==1, 2], 'bx')
    plt.show()
    