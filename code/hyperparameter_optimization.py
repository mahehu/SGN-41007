# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 07:59:23 2016

@author: hehu
"""


from sklearn.datasets import load_digits
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
import numpy as np
import scipy

def print_params(params, score):
    
    print "%.3f" % score,
    for key in params.keys():
        print key, "=", params[key], "/",
    print 
    
if __name__ == "__main__":
        
    digits = load_digits()
    
    numDigits = 10
    
    X = digits.data
    y = digits.target

    ########## Grid search with SVM
    # Specify 2 grids: for linear and rbf kernels

    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
         ]
         
    clf = GridSearchCV(SVC(), param_grid, cv=5,
                       scoring = 'accuracy')
    clf.fit(X, y)

    print("Scores:")
    scores = sorted(clf.grid_scores_, key = lambda x: x[1], reverse=True)

    for params, mean_score, scores in scores:
        print_params(params, mean_score) 
            
    # Random search
            
    # specify parameters and distributions to sample from
    param_grid = {"C": scipy.stats.expon(loc=0, scale = 5),
                  "kernel": ['linear', 'rbf'],
                  "gamma": scipy.stats.expon(loc = 0, scale = 0.1)}
    
    # run randomized search
    n_iter_search = 10
    
    clf = RandomizedSearchCV(SVC(), param_distributions=param_grid,
                                       n_iter=n_iter_search)

    clf.fit(X, y)
    
    print("Scores:")
    scores = sorted(clf.grid_scores_, key = lambda x: x[1], reverse=True)

    for params, mean_score, scores in scores:
        print_params(params, mean_score) 
        