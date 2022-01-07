# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:27:36 2015

@author: hehu
"""

import numpy as np
import os
from sklearn.linear_model import LogisticRegression

def load_arcene(path = "arcene"):

    X_train = np.loadtxt(path + os.sep + "arcene_train.data")
    y_train = np.loadtxt(path + os.sep + "arcene_train.labels")

    X_test = np.loadtxt(path + os.sep + "arcene_valid.data")
    y_test = np.loadtxt(path + os.sep + "arcene_valid.labels")

    # Split to training and testing at random
        
    return X_train, y_train, X_test, y_test
    
if __name__ == "__main__":

    # Read data:

    X_train, y_train, X_test, y_test = load_arcene()
    
    print "All data read."
    print "Result size is %s" % (str(X_train.shape))
    
    # Train a classifier
    clf = LogisticRegression()
    
    # Assess accuracy of the classifier with different parameters
    c_range = 10.0 ** np.arange(-10, -3)
    
    for C in c_range:
        clf.C = C
        clf.fit(X_train, y_train)
        
        predictions = clf.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        
        print "C = %.2e: accuracy = %.1f %%" % (C, 100.0 * accuracy)
        