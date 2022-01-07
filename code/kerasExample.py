# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 13:42:48 2016

@author: hehu
"""

from keras.models import Sequential
from keras.layers.core import Dense, Activation
import time
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def visualize(clf, X, y, X_new, 
              zoom = None, 
              zoom_scale = 2.5,
              annotate_prob = False, 
              plot_prob = False, 
              clabel = False):
    
    fig, ax = plt.subplots(figsize=[5,6])
        
    # create a mesh to plot in
    
    if clf is not None:

        h = .01  # step size in the mesh
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
    
        if plot_prob:
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)
            if clabel:
                levels = np.arange(0, 1.01, 0.1)
                CS = plt.contour(xx, yy, Z, levels = levels, hold = True)
                plt.contourf(xx, yy, Z, cmap='bwr', levels = levels, alpha=0.5)
                plt.clabel(CS, inline=1, fontsize=10)
                plt.colorbar()
            else:
                levels = np.arange(0, 1.01, 0.01)
                plt.contourf(xx, yy, Z, cmap='bwr', levels = levels, alpha=0.5)
                plt.colorbar()
                
        else:
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) > 0.5
            # Put the result into a color plot
        
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap='bwr', alpha=0.5) #plt.cm.Paired cmap='bwr',

    X1 = X[y==1, :]
    X2 = X[y==0, :]
    
    ax.plot(X1[:, 0], X1[:, 1], 'ro')
    ax.plot(X2[:, 0], X2[:, 1], 'bx')

    if X_new is not None:
        
        ax.plot([X_new[0]], [X_new[1]], 'ko')
    
        if clf is not None:
            yHat = "RED" if clf.predict(X_new).ravel() > 0.5 else "BLUE"
            
            if annotate_prob:
                prob = clf.predict_proba(X_new)
                annotation = "Classified as %s\np = %.1f %%" % (yHat, 100*np.max([prob, 1-prob]))
            else:
                annotation = "Classified as %s" % (yHat)
                
        else:
            annotation = "Which class?"
        
        ax.annotate(annotation, 
                 X_new, 
                 xytext=(4, 0.65), 
                 size=13,
                 bbox=dict(boxstyle="round4", fc="w", ec = "g"),
                 arrowprops=dict(arrowstyle="simple",
                                 connectionstyle="arc3,rad=-0.2",
                                 shrinkA = 0,
                                 shrinkB = 8,
                                 fc = "g",
                                 ec = "g"),
                 horizontalalignment='center', 
                 verticalalignment='middle')
    
    ax.set_ylim([-10, 2])
    ax.set_xlim([-2, 6])
    
    # Plot Zoom part

    if zoom is not None:
        #ax_zoom = zoomed_inset_axes(ax, 3, loc=4) # zoom = 3
        ax_zoom = zoomed_inset_axes(ax, zoom_scale, loc=4,
                     bbox_to_anchor=(0.915, -0.31),
                     bbox_transform=ax.figure.transFigure)
        ax_zoom.contourf(xx, yy, Z, cmap='bwr', alpha=0.5, interpolation="nearest",
             origin="lower")
    
        ax_zoom.plot(X1[:, 0], X1[:, 1], 'ro')
        ax_zoom.plot(X2[:, 0], X2[:, 1], 'bx')
    
        if X_new is not None:
            ax_zoom.plot(X_new[0], X_new[1], 'ko')
    
        # sub region of the original image
        x1, x2, y1, y2 = zoom
        ax_zoom.set_xlim(x1, x2)
        ax_zoom.set_ylim(y1, y2)
        
        plt.xticks(visible=False)
        plt.yticks(visible=False)

        mark_inset(ax, ax_zoom, loc1=1, loc2=3, fc="none", ec="0.0")
        
        plt.sca(ax)
        
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
    
    # Generate random training data 
        
    N = 200
    
    np.random.seed(2015)
    
    X1, X2 = generate_data(N)
    
    X = np.concatenate((X1.T, X2.T))
    y = np.concatenate((np.ones(N), np.zeros(N)))    
    
    clf = Sequential()
    
    clf.add(Dense(100, input_dim=2))
    clf.add(Activation('sigmoid'))
    
    clf.add(Dense(100))
    clf.add(Activation('sigmoid'))
    
    clf.add(Dense(1))
    clf.add(Activation('sigmoid'))
    
    startTime = time.time()
    clf.compile(loss='mean_squared_error', optimizer='sgd', metrics = ["accuracy"])
    print("Compilation takes %.2f s." % (time.time() - startTime))
    
    startTime = time.time()
    clf.fit(X, y, epochs=50, batch_size=16)
    print("Training takes %.2f s." % (time.time() - startTime))
    
    visualize(clf, X, y, None) #, zoom = [-0.5,1.5,-4,-2])
    plt.show()
    
    #plt.savefig("../images/mlp.pdf", bbox_inches = "tight", transparent = True)