# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 08:36:33 2019

@author: hehu
"""

import numpy as np
from scipy.signal import convolve2d 
import cv2
import matplotlib.pyplot as plt

x = cv2.imread("person1.jpg")
x = np.mean(x, axis = -1)

window_size = 5

w_left  = np.zeros(shape = (window_size, window_size // 2))
w_right = np.ones (shape = (window_size, window_size // 2 + 1))
w = np.concatenate((w_left, w_right), axis = 1)

w = w - np.mean(w)
y = convolve2d(x, w)

fig, ax = plt.subplots(1, 2, sharex = True, sharey = True)
ax[0].imshow(x, cmap = 'gray', interpolation = 'bilinear')
ax[1].imshow(y, interpolation = 'bilinear')

plt.show()
