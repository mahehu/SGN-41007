# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:57:30 2019

@author: hehu
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import time

digits = load_digits()
X = digits.data # shape = 1797 x 64
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = RandomForestClassifier(n_estimators = 100)

scores = cross_val_score(model, X, y)

start_time = time.time()
model.fit(X_train, y_train)
print("Elapsed time: {:.3f} s".format(time.time() - start_time))

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f} %".format(100 * accuracy))

importances = model.feature_importances_
importances = np.reshape(importances, (8,8))
plt.figure()
plt.imshow(importances)
plt.show()
