# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:05:43 2017

@author: heikki.huttunen@tut.fi

An example of using the LDA for pixel color classification.
Left mouse button shows examples of foreground, and
right mouse button examples of background.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D

from sklearn.neighbors import KNeighborsClassifier

def onclick(event):
    """
    Function that is run every time when used clicks the mouse.
    """
    
    ix, iy = event.xdata, event.ydata
    button = event.button
    
    if button == 2:
        # Stop when used clicks middle button (or kills window)
        fig.canvas.mpl_disconnect(cid)
        plt.close("all")
    else:
        # Otherwise add to the coords list.
        global coords
        coords.append([int(ix), int(iy), button])

if __name__ == "__main__":
        
    # Load test image and show it.
    
    img = plt.imread("hh.jpg")
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.set_title("Left-click: face; right-click: non-face; middle: exit")
    coords = []
    
    # Link mouse click to our function above.
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    X = []
    y = []
    
    for ix, iy, button in coords:
        
        # Collect nearby samples to the user clicked point
    
        w = img[iy-3 : iy+4, ix-3:ix+4, :]
        
        # Unnecessarily complicated line to collect color channels
        # into a matrix (results in 49x3 matrix)
        
        C = np.array([w[...,c].ravel() for c in [0,1,2]]).T
        X.append(C)
        
        # Store class information to y.
        
        if button == 1:
            y.append(np.ones(C.shape[0]))
        else:
            y.append(np.zeros(C.shape[0]))
            
    X = np.concatenate(X, axis = 0)
    y = np.concatenate(y, axis = 0)
    X_test = np.array([img[...,c].ravel() for c in [0,1,2]]).T
    
    # Switch between sklearn and our own implementation.
    # Don't know why these produce slightly different results.
    # Both seem to work though.
    
    use_sklearn = True
    
    if use_sklearn:
        #clf = LinearDiscriminantAnalysis()
        clf = KNeighborsClassifier()
        clf.fit(X, y)
        y_hat = clf.predict(X_test)
 
    else: # Do it the hard way:
        C0 = np.cov(X[y==0, :], rowvar = False)
        C1 = np.cov(X[y==1, :], rowvar = False)
        m0 = np.mean(X[y==0, :], axis = 0)
        m1 = np.mean(X[y==1, :], axis = 0)
        
        w = np.dot(np.linalg.inv(C0 + C1), (m1 - m0))
        T = 0.5 * (np.dot(w, m1) + np.dot(w, m0))    
 
        y_hat = np.dot(X_test, w) - T

        # Now y_hat is the class score.
        # Let's just threshold that at 0.
        # And cast from bool to integer
        
        y_hat = (y_hat > 0).astype(np.uint8)
    
    # Manipulate the vector form prediction to the original image shape.
    
    class_img = np.reshape(y_hat, img.shape[:2])

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img)
    ax[1].imshow(class_img)
    img[class_img == 0] = 0
    ax[2].imshow(img)

    # Show the data in a 3D plot.
    
    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    ax.plot(X[y==0, 0], X[y==0, 1], X[y==0, 2], 'ro')
    ax.plot(X[y==1, 0], X[y==1, 1], X[y==1, 2], 'bo')
    ax.plot(X_test[y_hat==0, 0], X_test[y_hat==0, 1], X_test[y_hat==0, 2], 'r.', alpha = 0.4)
    ax.plot(X_test[y_hat==1, 0], X_test[y_hat==1, 1], X_test[y_hat==1, 2], 'b.', alpha = 0.4)
  
    plt.show()
    