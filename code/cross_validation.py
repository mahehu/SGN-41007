# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 13:41:05 2016

@author: hehu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:01:16 2015

@author: hehu
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import pprint

from sklearn.cross_validation import cross_val_score, \
KFold, StratifiedShuffleSplit, \
LeaveOneOut

from scipy.linalg import eig
from scipy.stats import norm

M1 = np.array([[1.5151, -0.1129], [0.1399, 0.6287]])
M2 = np.array([[0.8602, 1.2461], [-0.0737, -1.5240]])
m1 = np.array([-1, 1])
m2 = np.array([-5, 2])

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
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
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
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
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])            
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
                
def logistic(x):
    y = 1 / (1 + np.exp(-x))
    return y
    
def show_hist(clf, X, y, logit = False):

    X1 = X[y==0, :]    
    X2 = X[y==1, :]    
    
    w = clf.coef_.T
    b = clf.intercept_

    X1_p = np.dot(X1, w) + b
    X2_p = np.dot(X2, w) + b
    X_p = np.dot(X, w) + b
    
    if logit:
        X1_p = logistic(X1_p)
        X2_p = logistic(X2_p)
        X_p = logistic(X_p)
        
    proj_min = np.min(X_p)
    proj_max = np.max(X_p)

    bins = np.linspace(proj_min, proj_max, 40)
    plt.hist(X1_p, bins = bins, color = 'red', alpha = 0.5)
    plt.hist(X2_p, bins = bins, color = 'blue', alpha = 0.5)    

def generate_data(N):

    global M1
    global M2
        
    X1 = np.random.randn(N, 2)
    X2 = np.random.randn(N, 2)
    
    X1 = np.matmul(X1, M1)
    X2 = np.matmul(X2, M2)
    
    T1 = np.array(m1).reshape((1,2))
    T2 = np.array(m2).reshape((1,2))
    
    X1 = X1 + np.tile(T1, [N, 1])
    X2 = X2 + np.tile(T2, [N, 1])

    X1 = X1[::-1,:]    
    X2 = X2[::-1,:]
    
    return X1, X2

def get_true_accuracy(clf):
    
    a = clf.coef_.ravel()
    b = clf.intercept_.ravel()
    
    global M1
    global M2
    
    global m1
    global m2
    
    C1 = np.matmul(M1.T, M1)
    C2 = np.matmul(M2.T, M2)
    
    var1 = np.dot(np.dot(a, C1), a)
    var2 = np.dot(np.dot(a, C2), a)
    
    mean1 = np.dot(a, m1)
    mean2 = np.dot(a, m2)
    
    A1 = norm.cdf(b, loc = mean1, scale = np.sqrt(var1)).ravel()
    A2 = norm.sf(b, loc = mean2, scale = np.sqrt(var2)).ravel()
    
    return (A1 + A2)[0]

def get_toy_data(N):
    
    X1, X2 = generate_data(N)
    
    X = np.concatenate((X1, X2))
    y = np.concatenate((np.ones(N), np.zeros(N)))  
    
    return X, y
    
if __name__ == "__main__":
        
    plt.close("all")
    
    # Generate random training data 
        
    N = 200
    
    np.random.seed(2015)
    
    X, y = get_toy_data(N)
        
    clf = LogisticRegression()

    
    scores = cross_val_score(clf, X, y, cv = 10)
    print ("Accuracy: %.2f +- %.2f" %
          (np.mean(scores), 
           np.std(scores)))
    
    loo = LeaveOneOut(y.size)
    scores = cross_val_score(clf, X, y, cv = loo)
    print ("Accuracy: %.2f +- %.2f" %
          (np.mean(scores), 
           np.std(scores)))
    
    metrics = ['accuracy',
               'roc_auc',
               'recall',
               'precision',
               'f1']
    
    for scorer in metrics:
        scores = cross_val_score(clf, 
                                 X, 
                                 y, 
                                 scoring = scorer)
                                 
        print ("%s score: %.2f +- %.2f" % \
              (scorer,
               np.mean(scores),
               np.std(scores)))

    K_values = np.linspace(2, 400, 100).astype(int)
    K_scores = []
    K_vars = []
        
    for K in K_values:
        differences = []
        for seed in range(100):
            np.random.seed(seed)
            X, y = get_toy_data(N = 200)
            kf = KFold(X.shape[0], n_folds = K, random_state = seed, shuffle = True)
            
            accuracies = []
            
            for train_idx, test_idx in kf:
                clf.fit(X[train_idx, :], y[train_idx])
                y_pred = clf.predict(X[test_idx])
                scores = accuracy_score(y[test_idx], y_pred)
                accuracies.append(np.mean(scores))
            
            clf.fit(X, y)
            
            cv_accuracy = np.mean(accuracies)
            true_accuracy = get_true_accuracy(clf)
            
            differences.append(true_accuracy - cv_accuracy)
            
        K_scores.append(np.mean(np.abs(differences)))
        K_vars.append(np.var(differences))
        
        print("K =", K, "done", K_scores[-1])
        
    fig, ax1 = plt.subplots()
    ax1.plot(K_values, K_scores, 'ro-', linewidth =2) 
    ax1.set_ylabel('Deviation from true error')
    
    plt.savefig("../images/K_scores.pdf", bbox_inches = "tight")
    
    ax1.set_ylabel('Mean deviation from true error', color='r')
    ax1.tick_params('y', colors='r')
    ax2 = ax1.twinx()
    
    ax2.plot(K_values, K_vars, 'bo-', linewidth =2)
    ax2.set_ylabel('Variance of CV estimates', color='b')
    ax2.tick_params('y', colors='b')
    
    ax1.set_xlabel("Number of folds")
    #plt.grid()
    #plt.show()
    plt.savefig("../images/K_scores_var.pdf", bbox_inches = "tight")
    