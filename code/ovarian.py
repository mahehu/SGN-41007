# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:42:02 2015

@author: hehu
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from scipy.signal import lfilter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from random_forest import load_arcene
from sklearn.preprocessing import Normalizer
import copy

if __name__ == "__main__":
    
    # Load data
    
    np.random.seed(1)
    
    X_train, y_train, X_test, y_test = load_arcene()

    smooth = True
    N = 51.0
    if smooth:

        X_train = lfilter(np.ones(N) / N, [1], X_train, axis = 1)
        X_test  = lfilter(np.ones(N) / N, [1], X_test, axis = 1)
        healthy = X_train[y_train==-1, :][0]
        ovarian = X_train[y_train==1, :][0]
        
    else:
        healthy = X_train[y_train==-1, :][0]
        ovarian = X_train[y_train==1, :][0]
        healthy = lfilter(np.ones(N)/N, [1], healthy)
        ovarian = lfilter(np.ones(N)/N, [1], ovarian)
        
    normalizer = Normalizer()
    normalizer.fit(X_train)        
    X_train = normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)
    
    plt.figure(figsize=(10,3))

    plt.plot(healthy, 'b-', label = "Healthy")
    plt.plot(ovarian, 'r-', label = "Cancer")
    plt.legend()
    plt.title("Proteomic Mass Spectrum (Smoothed)")
    plt.savefig("../images/ovariancancer_plot.pdf", bbox_inches = "tight")
    
    # Train a set of classifiers with different C:
    
    classifiers = [(LogisticRegression(), "LogReg"),
                   (LinearSVC(dual = False), "SVM")]
    C_range = 10.0 ** np.arange(0,12, 0.25)
    
    bestAccuracy = 0
    
    for clf, name in classifiers:

        for penalty in ["l1", "l2"]:            
            clf.penalty = penalty
            
            accuracies = []
            nonzeros = []
            
            for C in C_range:
                
                clf.C = C
                clf.fit(X_train, y_train)
                print "%d nonzeros" % (np.count_nonzero(clf.coef_))
                # See how good we are:
                prediction = clf.predict(X_test)
                accuracy = 100.0 * np.mean(prediction == y_test)
                print "%s (penalty = %s, C = %.2e): Classification Accuracy: %.2f %%" % (name, penalty, C, accuracy)
                accuracies.append(accuracy)
                nonzeros.append(np.count_nonzero(clf.coef_))
                
                if accuracy > bestAccuracy:
                    bestAccuracy = accuracy
                    bestClf = copy.deepcopy(clf)
                    bestC = C
                    bestName = name
                    
            fig, ax1 = plt.subplots()
            ax1.semilogx(C_range, accuracies, 'b-', linewidth = 2)
            ax1.set_ylabel("Accuracy / %", color = "b")
            for tl in ax1.get_yticklabels():
                tl.set_color('b')
            ax1.set_ylim(0, 100)
            ax2 = ax1.twinx()
            ax2.semilogx(C_range, nonzeros, 'r-', linewidth = 2)
            ax2.set_ylabel("Number of Nonzeros", color = "r")
            for tl in ax2.get_yticklabels():
                tl.set_color('r')
            ax2.set_ylim(0, 10000)
            plt.xlabel("C")
            plt.title("%s Classifier (%s penalty)" % (name, penalty.upper()))
            
            maxIdx = np.argmax(accuracies)
            maxC = C_range[maxIdx]
            maxAcc = accuracies[maxIdx]
            maxNonzeros = nonzeros[maxIdx]
            
            ax1.annotate("Peak Accuracy %.1f %%" % maxAcc, 
                        (maxC, maxAcc), 
                        xytext=(1e8, 10), 
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

            ax2.annotate("%d Nonzeros" % maxNonzeros, 
                        (maxC, maxNonzeros), 
                        xytext=(1e2, 3000), 
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
                        
            ax1.plot([maxC, maxC],[0,100], 'k--')
            ax1.plot([maxC], [maxAcc], 'go')
            ax2.plot([maxC], [maxNonzeros], 'go')
            plt.tight_layout()
            
            plt.savefig("../images/ovarian_%s_%s.pdf" % (name, penalty), bbox_inches = "tight")
            
    plt.figure(figsize = [6,4])
    plt.stem(bestClf.coef_.ravel()) 
    nnz = np.count_nonzero(bestClf.coef_.ravel())
    plt.title("Coefficients of %s with C = %.3f (%d nonzero)" % (bestName, bestC, nnz))
    plt.savefig("../images/ovarian_coef.pdf", bbox_inches = "tight")
    