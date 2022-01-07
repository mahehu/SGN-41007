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
from sklearn.cluster import KMeans

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
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
        
    #plt.style.use('classic')
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

    kmeans = KMeans(n_clusters = 2)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    
    figsize = [5,4]
    
    plt.figure(figsize = figsize)
    plt.plot(X[:, 0], X[:, 1], 'gs')
    plt.title("Original unlabeled data")
    plt.grid('on')
    plt.savefig("../images/kmeans_orig.pdf", bbox_inches = "tight")
    
    styles = ['bx', 'ro', 'c+', 'y*', 'mh', 'gd']
    
    for K in range(2, 6):
        kmeans = KMeans(n_clusters = K)
        kmeans.fit(X)
        y_pred = kmeans.predict(X)
        
        plt.figure(figsize = figsize)
        for k in range(K):
            plt.plot(X[y_pred == k, 0], X[y_pred == k, 1], styles[k], label="Cluster %d" % (k+1))
            
        plt.title("K-Means split to %d clusters" % K)
        plt.legend()
        plt.grid('on')
        plt.savefig("../images/kmeans_%d.pdf" % K, bbox_inches = "tight")
        
    plt.figure(figsize = figsize)
    for k in range(2):
        plt.plot(X[y == k, 0], X[y == k, 1], styles[k], label="Class %d" % (k+1))
        
    plt.title("Ground Truth")
    plt.legend()
    plt.grid('on')
    plt.savefig("../images/kmeans_true.pdf", bbox_inches = "tight")
