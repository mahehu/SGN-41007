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
    
    # Train a Random forest classifier witn 100 trees, and plot the feature importances.
    