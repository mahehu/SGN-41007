# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:25:48 2016

@author: hehu
"""

import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_covtype
import matplotlib.pyplot as plt
import matplotlib
import time
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.lda import LDA

def load_arcene(path = "arcene"):

    X_train = np.loadtxt(path + os.sep + "arcene_train.data")
    y_train = np.loadtxt(path + os.sep + "arcene_train.labels")

    X_test = np.loadtxt(path + os.sep + "arcene_valid.data")
    y_test = np.loadtxt(path + os.sep + "arcene_valid.labels")

    # Split to training and testing at random
        
    return X_train, y_train, X_test, y_test
    
if __name__ == "__main__":
    
    matplotlib.rcdefaults()
    font = {'family' : 'sans-serif'}
    matplotlib.rc('font', **font)
    
    X_train, y_train, X_test, y_test = load_arcene()

    numIters = 10
    numEstimators = 10
        
    ###################################

    classifiers = [(LDA(), "LDA"),
                   (LinearSVC(), "Linear SVM"),
                   (SVC(), "SVM with RBF Kernel"),
                   (LogisticRegression(), "Logistic Regression")]
               
    f, ax = plt.subplots(len(classifiers), figsize = [5,12], sharex=True)
    
    for i, (clf, name) in enumerate(classifiers):
        accuracies = []
        print "Training %s..." % name
        

        startTime = time.time()
        
        for iteration in range(numIters):
            
            clf.fit(X_train, y_train)
            
            y_hat= clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_hat)
            
            accuracies.append(accuracy)
    
        elapsedTime = time.time() - startTime
        bins = np.arange(0, 1.01, 0.05)
        
        ax[i].hist(accuracies, bins)
        ymin, ymax = ax[i].get_ylim()
        ax[i].set_title("%s: accuracy = %.2f$\pm$%.2f" % (name, np.mean(accuracies), np.std(accuracies)))
        ax[i].text(0.05, 0.85*ymax + 0.15*ymin, "%.2f sec / iteration" % (elapsedTime / numIters),
                   style='italic',
                   bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
        
        plt.savefig("../images/LinearComparison.pdf", bbox_inches = "tight", transparent = True)
    
    ###################################
    
    classifiers = [(RandomForestClassifier(), "Random Forest"),
                   (ExtraTreesClassifier(), "Extra-Trees"),
                   (AdaBoostClassifier(), "AdaBoost"),
                   (GradientBoostingClassifier(), "GB-Trees")]
               
    f, ax = plt.subplots(len(classifiers), figsize = [5,12], sharex=True)
    
    for i, (clf, name) in enumerate(classifiers):
        
        clf.n_estimators = numEstimators

        accuracies = []
        print "Training %s..." % name
        
        startTime = time.time()
        
        for iteration in range(numIters):
            
            clf.n_estimators = numEstimators
            clf.fit(X_train, y_train)
            
            y_hat= clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_hat)
            
            accuracies.append(accuracy)
    
        elapsedTime = time.time() - startTime
        bins = np.arange(0, 1.01, 0.05)
        
        ax[i].hist(accuracies, bins)
        ymin, ymax = ax[i].get_ylim()
        ax[i].set_title("%s: accuracy = %.2f$\pm$%.2f" % (name, np.mean(accuracies), np.std(accuracies)))
        ax[i].text(0.05, 0.85*ymax + 0.15*ymin, "%.2f sec / iteration" % (elapsedTime / numIters),
                   style='italic',
                   bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
        
        plt.savefig("../images/EnsembleComparison.pdf", bbox_inches = "tight", transparent = True)
    

    ###################################
    
    classifiers = [(KNeighborsClassifier(n_neighbors = 1), "1-Nearest-Neighbor"),
                   (KNeighborsClassifier(n_neighbors = 5), "5-Nearest-Neighbor"),
                   (KNeighborsClassifier(n_neighbors = 9), "9-Nearest-Neighbor")]
               
    f, ax = plt.subplots(nrows = 1, ncols = len(classifiers), figsize = [18,3], sharex=True)
    
    for i, (clf, name) in enumerate(classifiers):
        
        clf.n_estimators = numEstimators

        accuracies = []
        print "Training %s..." % name
        
        startTime = time.time()
        
        for iteration in range(numIters):
            
            clf.fit(X_train, y_train)
            
            y_hat= clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_hat)
            
            accuracies.append(accuracy)
    
        elapsedTime = time.time() - startTime
        bins = np.arange(0, 1.01, 0.05)
        
        ax[i].hist(accuracies, bins)
        ymin, ymax = ax[i].get_ylim()
        ax[i].set_title("%s: accuracy = %.2f$\pm$%.2f" % (name, np.mean(accuracies), np.std(accuracies)))
        ax[i].text(0.05, 0.85*ymax + 0.15*ymin, "%.2f sec / iteration" % (elapsedTime / numIters),
                   style='italic',
                   bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
        
        plt.savefig("../images/KnnComparison.pdf", bbox_inches = "tight", transparent = True)
    
