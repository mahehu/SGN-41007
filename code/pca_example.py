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

from sklearn.model_selection import cross_val_score, \
KFold, StratifiedShuffleSplit, \
LeaveOneOut

from scipy.linalg import eig
from scipy.stats import norm

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
    
    X = X - np.tile(X.mean(axis = 0), [X.shape[0], 1])
    
    plt.figure(figsize = [5,8])
    plt.plot(X[:, 0], X[:,1], 'ro')
    #plt.plot(plt.xlim(), [0,0], 'k-')
    #plt.plot([0,0], plt.ylim(), 'k-')
    plt.axis('tight')
    
    #D, W = np.linalg.eig(np.cov(X, rowvar = False))
    D, W = np.linalg.eig(np.matmul(X.T, X))
    W = W[:, np.argsort(D)[::-1]]
    d = -5
    plt.arrow(0, 0, d * W[0,0], d * W[1,0], zorder = 3, fc = 'k', width = 0.2, head_width=0.6, head_length=0.3)
    plt.arrow(0, 0, d * W[0,1], d * W[1,1], zorder = 3, fc = 'k', width = 0.2, head_width=0.6, head_length=0.3)
    plt.grid()
    
    plt.axis('equal')
    
    plt.annotate("First PC", 
                         0.9 * d * W[:, 0], 
                         xytext=(-4, 5), 
                         size=13,
                         bbox=dict(boxstyle="round4", fc="w", ec = "g"),
                         arrowprops=dict(arrowstyle="simple",
                                         connectionstyle="arc3,rad=0.2",
                                         shrinkA = 0,
                                         shrinkB = 8,
                                         fc = "g",
                                         ec = "g"),
                         horizontalalignment='center', 
                         verticalalignment='baseline')
    plt.annotate("Second PC", 
                         0.7 * d * W[:, 1], 
                         xytext=(4, -1), 
                         size=13,
                         bbox=dict(boxstyle="round4", fc="w", ec = "g"),
                         arrowprops=dict(arrowstyle="simple",
                                         connectionstyle="arc3,rad=0.2",
                                         shrinkA = 0,
                                         shrinkB = 8,
                                         fc = "g",
                                         ec = "g"),
                         horizontalalignment='center', 
                         verticalalignment='baseline')
                         
    plt.savefig("../images/PCA_example.pdf", bbox_inches = "tight")
    
    print("First PC: %s" % str(-W[:, 0]))
    print("Second PC: %s" % str(-W[:, 1]))
    
    ######## Digits example
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import Normalizer
    
    digits = load_digits()
    X = digits.data
    y = digits.target

    X = X - np.tile(X.mean(axis = 0), [X.shape[0], 1])
    D, V = np.linalg.eig(np.cov(X, rowvar = False))

    X_rot = np.matmul(X, V)
    
    fig, ax = plt.subplots(1,2,figsize = [5,3])
    
    ax[0].imshow(digits.images[0], cmap = 'gray', interpolation = 'bilinear')
    ax[0].set_title("Original")
    ax[1].imshow(X_rot[0, :].reshape(8,8), interpolation = 'nearest')
    ax[1].set_title("PCA mapped")
    
    plt.savefig('../images/digits_pca_proj_example1.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1,2,figsize = [5,3])
    
    ax[0].imshow(digits.images[800], cmap = 'gray', interpolation = 'bilinear')
    ax[0].set_title("Original")
    ax[1].imshow(X_rot[800, :].reshape(8,8), interpolation = 'nearest')
    ax[1].set_title("PCA mapped")
    
    plt.savefig('../images/digits_pca_proj_example2.pdf', bbox_inches='tight')
    
    plt.figure()
    plt.bar(np.arange(X_rot.shape[1]), np.var(X_rot, axis = 0))
    plt.grid()
    plt.axis('tight')    
    plt.title("Variances of principal components")
    plt.savefig('../images/digits_pca_proj_vars.pdf', bbox_inches='tight')
    
    plt.figure(figsize = [5,8])
    labels = ['zeros', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixes', 'sevens', 'eights', 'nines']
    
    for k in range(10):
        plt.scatter(X_rot[y == k, 0], X_rot[y == k, 1], label = labels[k])
    
    plt.legend(loc = 'best')
    plt.savefig('../images/digits_pca_proj.pdf', bbox_inches='tight')
    
    fig, ax = plt.subplots(4, 2, figsize = [3,7])
    
    for k in range(4):
        for j in range(2):
            ax[k][j].imshow(V[:, k*2 + j].reshape(8,8), interpolation = 'bilinear')
            ax[k][j].set_title("Eigenimage %d" % (k*2 + j + 1))

    plt.tight_layout()
    plt.savefig('../images/digits_eigenimages.pdf', bbox_inches='tight')
    
    #T-SNE

    ######## Digits example

    from sklearn.datasets import load_digits
    from sklearn.preprocessing import Normalizer
    from sklearn.manifold import TSNE
    
    digits = load_digits()
    X = digits.data
    y = digits.target

    X = X - np.tile(X.mean(axis = 0), [X.shape[0], 1])

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize = [5,8])
    labels = ['zeros', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixes', 'sevens', 'eights', 'nines']
    
    for k in range(10):
        plt.scatter(X_tsne[y == k, 0], X_tsne[y == k, 1], label = labels[k])
    
    plt.legend(loc = 'best')
    plt.savefig('../images/digits_tsne_proj.pdf', bbox_inches='tight')
    
    
    #####################
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import accuracy_score
    from scipy.io import loadmat
    from sklearn.decomposition import PCA
    
    D = loadmat("arcene.mat")
   
    X_test = D["X_test"]
    X_train = D["X_train"]
    y_test = D["y_test"].ravel()
    y_train = D["y_train"].ravel()
    	
    normalizer = Normalizer()
    normalizer.fit(X_train)
    X_train = normalizer.transform(X_train)
    X_test  = normalizer.transform(X_test)	
    
    pca = PCA()
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test  = pca.transform(X_test)

    accuracies = []
    
    for num_components in range(1, 99):
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train[:, :num_components], y_train)    
        y_pred = clf.predict(X_test[:, :num_components])
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    
    plt.figure()
    plt.plot(accuracies, linewidth = 2)
    
    topScore = np.max(accuracies)
    topIdx = np.argmax(accuracies)
    
    plt.annotate('Top score %.1f %%' % (100.0*topScore), 
             xy=(topIdx + 1, topScore), 
             xytext=(50, 0.65), 
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
    
    plt.annotate('100 components %.1f %%' % (100.0*accuracies[-1]), 
             xy=(99, accuracies[-1]), 
             xytext=(70, 0.6), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=0.2",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='middle')
             
    plt.grid()
    plt.title("Classification accuracy")
    plt.xlabel("Number of PCA components")
    plt.ylabel("Accuracy / %")
    plt.savefig('../images/arcene_pca.pdf', bbox_inches='tight')
    