# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 08:52:59 2015

@author: hehu
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from numpy import pi

def get_obs_matrix(x, order = 5):
    """
    Return the observation matrix 
    constructed from powers of vector x.
    """
    
    H = []

    for k in range(order):
        H.append(x**k)
        
    H = np.array(H).T
    
    return H

if __name__ == "__main__":
    
    # generate a noisy sinusoid
    
    np.random.seed(2015)
    
    x = np.arange(0,1.01,0.08)
    y = np.cos(2 * 2*pi*x) \
        + 0.35*np.random.randn(x.shape[0])
    y2 = np.cos(1 * 2*pi*x) \
        + 0.35*np.random.randn(x.shape[0])
    x = np.arange(0,1.01,0.04)
    
    y = np.concatenate((y, y2))
    
    plt.figure(figsize=(5,3))
    plt.plot(x, y, 'ro-')
    plt.title("Example time series")
    plt.axis([0,1,-3,3])
        
    y = np.atleast_2d(y).T
    
    plt.savefig("../images/timeSeries.pdf", bbox_inches='tight')
    
    for order in [4, 7, 25]:
        H = get_obs_matrix(x, order = order)
        
        plt.figure(figsize=(5,3))
        
        model = Ridge(alpha = 0)
        model.fit(H, y)
        x2 = np.arange(0,1.01,0.001)
        yHat = model.predict(get_obs_matrix(x2, order = order))
        
        plt.plot(x,y,'ro')
        plt.plot(x2, yHat, 'g-', linewidth=3)
        plt.axis([0,1,-3,3])
        
        plt.title("%d'th order polynomial fit" % order)
        plt.savefig("../images/timeSeries_%d.pdf" % order, bbox_inches='tight')
        
    
    for order in [4, 7, 25]:
        H = get_obs_matrix(x, order = order)
        
        plt.figure(figsize=(5,3))
        
        model = Ridge(alpha = 1e-12)
        model.fit(H, y)
        x2 = np.arange(0,1.01,0.001)
        yHat = model.predict(get_obs_matrix(x2, order = order))
        
        plt.plot(x,y,'ro')
        plt.plot(x2, yHat, 'g-', linewidth=3)
        plt.axis([0,1,-3,3])
        
        plt.title("%d'th order regularized polynomial fit" % order)
        plt.savefig("../images/timeSeries_%d_reg.pdf" % order, bbox_inches='tight')

    order = 25
    H = get_obs_matrix(x, order = order)

    model = Ridge(alpha = 0)
    model.fit(H, y)
    
    unreg_coef = model.coef_.ravel()
    
    plt.figure(figsize=(5,3))
    plt.plot(x,y,'ro')
    y2 = model.predict(get_obs_matrix(x2, order = order)).ravel()
    plt.plot(x2, y2, 'g-', linewidth=3)
    plt.axis([0,1,-3,3])
    plt.title("Unregularized Fit")    
    plt.savefig("../images/unreg_fit.pdf") 
    
    model = Ridge(alpha = 1e-3)
    model.fit(H, y)
    ridge_coef = model.coef_.ravel()
    yHat_L2 = model.predict(get_obs_matrix(x2, order = order))
    
    
    plt.figure(figsize=(5,3))
    plt.plot(x,y,'ro')
    y2 = model.predict(get_obs_matrix(x2, order = order)).ravel()
    plt.plot(x2, y2, 'g-', linewidth=3)
    plt.axis([0,1,-3,3])
    plt.title("L2 Regularized Fit")    
    plt.savefig("../images/L2_fit.pdf") 


    model = Lasso(alpha = 1e-3)
    model.fit(H, y)
    lasso_coef = model.coef_.ravel()
    yHat_L1 = model.predict(get_obs_matrix(x2, order = order))
    
    plt.figure(figsize=(5,3))
    plt.plot(x,y,'ro')
    y2 = model.predict(get_obs_matrix(x2, order = order)).ravel()
    plt.plot(x2, y2, 'g-', linewidth=3)
    plt.axis([0,1,-3,3])
    plt.title("L1 Regularized Fit")    
    plt.savefig("../images/L1_fit.pdf") 
    
    plt.figure(figsize=(5,3))
    plt.stem(unreg_coef)
    plt.title("Unregularized model: %d nonzeros" % np.count_nonzero(unreg_coef))    
    plt.xlabel("Order")
    plt.savefig("../images/unreg_coef.pdf")        

    plt.figure(figsize=(5,3))
    plt.stem(ridge_coef)
    plt.title("$L_2$ penalty: %d nonzeros" % np.count_nonzero(ridge_coef))    
    plt.xlabel("Order")
    plt.savefig("../images/RR_coef.pdf")
        
#    plt.figure(figsize=(5,3))
#    plt.plot(x,y,'ro')
#    plt.plot(x2, yHat_L2, 'g-', linewidth=3)
#    plt.axis([0,1,-3,3])
#    plt.title("L2 regularized model")    

    plt.figure(figsize=(5,3))
    plt.stem(lasso_coef)
    plt.title("$L_1$ penalty: %d nonzeros" % np.count_nonzero(lasso_coef))    
    plt.xlabel("Order")
    plt.savefig("../images/lasso_coef.pdf")
    
#    plt.figure(figsize=(5,3))
#    plt.plot(x,y,'ro')
#    plt.plot(x2, yHat_L1, 'g-', linewidth=3)
#    plt.axis([0,1,-3,3])
#    plt.title("L1 regularized model")    
