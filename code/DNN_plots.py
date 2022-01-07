# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:01:16 2015

@author: hehu
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import matplotlib

from scipy.linalg import eig

if __name__ == "__main__":
    
    x = np.linspace(-5,5)
    y = 0.05 + x * (x > 0)
    plt.subplot(211)
    plt.plot(x,y, linewidth=3, label = 'ReLU function')
    plt.axis([-5,5,0,5])
    plt.legend(loc = 'best')
    
    plt.savefig("../images/ReLU.pdf", bbox_inches = "tight", transparent = True)    
    