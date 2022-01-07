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
        
    X_train, y_train, X_test, y_test = load_arcene()

    numEstimators = 100
        
    clf = RandomForestClassifier(n_estimators = numEstimators)
    clf.fit(X_train, y_train)
    
    #plt.bar(range(100), sorted(clf.feature_importances_)[-100:])
    plt.plot (clf.feature_importances_)