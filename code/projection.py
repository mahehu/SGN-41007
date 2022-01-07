# -*- coding: utf-8 -*-
"""
Created on Tue Feb 09 10:14:28 2016

@author: hehu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal

m0 = np.array([-5, 5])
m1 = np.array([10, 5])

R = np.array([[1, 2],[0, 1]])
S = np.array([[14, 0],[0, 1]])
C0 = np.dot(R, np.dot(S, R.T))

R = np.array([[1, 0],[1, 1]])
S = np.array([[1, 0],[0, 50]])
C1 = np.dot(R, np.dot(S, R.T))

print C0
print C1
print m0
print m1

def generate_data(N):

    X0 = multivariate_normal(m0, C0).rvs(N)
    X1 = multivariate_normal(m1, C1).rvs(N)

    return X0, X1

def get_toy_data(N):
    
    X0, X1 = generate_data(N)
    
    X = np.concatenate((X0, X1), axis = 0)
    y = np.concatenate((np.zeros(N), np.ones(N)))  
    
    return X, y
    
if __name__ == "__main__":

    plt.close("all")
    N = 500    
    X, y = get_toy_data(N)
    
    plt.figure()
    plt.plot(X[y==0, 0], X[y==0, 1], 'ro', label = "Class 0")
    plt.plot(X[y==1, 0], X[y==1, 1], 'bx', label = "Class 1")
    plt.legend(loc = "best")
    plt.savefig("../../../Tentit/SGN-41006/2classes_09.pdf", bbox_inches = "tight")

    w = np.dot(np.linalg.inv(C0 + C1), (m1 - m0))
    
    m0 = np.dot(w, m0)
    m1 = np.dot(w, m1)
    s0 = np.dot(np.dot(w, C0), w)
    s1 = np.dot(np.dot(w, C1), w)
    
    x = np.linspace(np.dot(X, w).min(), np.dot(X, w).max(),500)
    g0 = norm.pdf(x, m0, np.sqrt(s0))
    g1 = norm.pdf(x, m1, np.sqrt(s1))
    plt.figure()
    plt.plot(x, g0, 'r-', label = "Class 0", linewidth = 2)    
    plt.plot(x, g1, 'b-', label = "Class 1", linewidth = 2)    
    plt.legend(loc = "best")
    
    idx0 = np.argmin(np.abs(x - m0))
    idx1 = np.argmin(np.abs(x - m1))
    x0 = x[idx0:idx1]
    
    c = x0[np.argmin(np.abs(g1[idx0:idx1] - g0[idx0:idx1]))]
    ymin, ymax = plt.gca().get_ylim()    
    plt.plot([c, c], [ymin, ymax], 'k--')
    plt.title("Threshold = %.3f" % c)
    
    print c - m0
    print m1 - c
    
    plt.figure()
    h, b = np.histogram(np.dot(X[y==0,:], w), bins=100)
    plt.plot(b[:-1], h, 'r', alpha = 0.5)
    h, b = np.histogram(np.dot(X[y==1,:], w), bins=100)
    plt.plot(b[:-1], h, 'b', alpha = 0.5)
    