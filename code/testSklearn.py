# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:47:17 2015

@author: hehu
"""

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

if __name__ == "__main__":

    # Test basic sklearn functionality
    X = np.random.rand(1000, 5)
    y = (np.random.rand(1000,) + 0.5).astype(int)
    X[y==1,:] += 0.5
    
    clf = LogisticRegression()

    scores = cross_val_score(clf, X, y, cv = 5)
    
    # Test building a Keras model and training it
    model = Sequential()
    
    model.add(Dense(output_dim=1, input_dim=5))
    model.add(Activation('sigmoid'))
    
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    model.fit(X[:900, :], y[:900], nb_epoch=200, batch_size=16)
    yHat = np.round(model.predict(X[900:, :]).ravel())
    accuracy = np.mean(yHat == y[900:])
    
    print "Sklearn model trained. Accuracy: %.2f" % (np.mean(scores))
    print "Keras model trained. Accuracy: %.2f" % (accuracy)
    