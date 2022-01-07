# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 11:05:19 2018

@author: hehu
"""

import numpy as np
from scipy.signal import convolve2d 
import cv2
import matplotlib.pyplot as plt

x = cv2.imread("person1.jpg")
x = np.mean(x, axis = -1)

w = np.array([[0,1,1], [0,1,1], [0,1,1]])
w = w - np.mean(w)
y = convolve2d(x, w)

fig, ax = plt.subplots(1, 2, sharex = True, sharey = True)
ax[0].imshow(x, cmap = 'gray')
ax[1].imshow(y, cmap = 'gray')

plt.show()
