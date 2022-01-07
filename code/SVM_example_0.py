# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:05:52 2016

@author: hehu
"""

from sklearn import svm
import cv2

X = [[0, 0], [1, 1]]
y = [0, 1]

clf = svm.SVC()
clf.fit(X, y)  
