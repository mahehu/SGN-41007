# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:57:30 2019

@author: hehu
"""

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data # shape = 1797 x 64
y = digits.target

tree = DecisionTreeClassifier()
tree.fit(X, y)
plot_tree(tree)  

plt.savefig("tree.pdf", bbox_inches = "tight")
