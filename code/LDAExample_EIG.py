# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:50:30 2018

@author: hehu
"""


import matplotlib.pyplot as plt
import numpy as np


from scipy.linalg import eig


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
    
    X1 = X1.T
    X2 = X2.T
    X1_test = X1_test.T
    X2_test = X2_test.T
        
    X_test = np.concatenate((X1_test, X2_test))
    y_test = np.concatenate((np.ones(N), np.zeros(N)))    

    m1 = np.mean(X1, axis = 0)
    m2 = np.mean(X2, axis = 0)
    C1 = np.cov(X1 - m1, rowvar = False)
    C2 = np.cov(X2 - m2, rowvar = False)
    
    SB = np.multiply.outer(m2-m1, m2-m1)
    SW = C2 + C1

    D, V = eig(SB, SW)
    w = V[:,1]
    T = np.mean([np.dot(m1, w), np.dot(m2, w)])
    
    print(w)
    
    
    