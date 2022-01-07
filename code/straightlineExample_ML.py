# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:58:23 2015

@author: hehu
"""


import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    
    np.random.seed(2016)
    
    x = np.linspace(-10,10,100)
    y = 1 / (1 + np.exp(-x))
    
    x = x + 0.1 * np.random.randn(x.size)
    y = y + 0.1 * np.random.randn(y.size)

    fig, ax = plt.subplots(figsize=[5,6])
    
    ax.plot(x, y, 'ro')
    
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.savefig("../images/LSEx1Data.pdf")
    
    np.save("x.npy", x)
    np.save("y.npy", y)
    
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    sum_squares = np.sum((m*x + c - y)**2)

    plt.plot(x, m*x + c, 'b-', label='Model candidate 1\n(a = %.2f, b = %.2f)' % (m, c), linewidth=2.0)
    
    m2 = m + 0.01*np.random.randn()    
    c2 = c + 0.1 * np.random.randn()
    plt.plot(x, m2*x + c2, 'g-', label='Model candidate 2\n(a = %.2f, b = %.2f)' % (m2, c2), linewidth=2.0)
    
    m2 = m + 0.01*np.random.randn()    
    c2 = c + 0.1 * np.random.randn()
    plt.plot(x, m2*x + c2, 'm-', label='Model candidate 3\n(a = %.2f, b = %.2f)' % (m2, c2), linewidth=2.0)
    
    plt.legend(loc = 2)
    
    plt.savefig("../images/LSEx1.pdf")
    
    fig, ax = plt.subplots(figsize=[5,6])
    
    ax.plot(x, y, 'ro')
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, m*x + c, 'b-', label='Best Fit\ny = %.4fx+%.4f\nSum of Squares = %.2f' % (m, c, sum_squares), linewidth=2.0)
    for k in range(len(x)):
        x0 = x[k]
        y0 = y[k]
        x1 = x0
        y1 = m*x0 + c
        plt.plot([x0, x1], [y0, y1], 'g--', alpha = 0.5)
        plt.legend(loc = 2)
        
    plt.savefig("../images/LSEx1Solution.pdf")
    