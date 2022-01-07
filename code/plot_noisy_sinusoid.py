# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 11:55:15 2017

@author: hehu
"""

import numpy as np
import matplotlib.pyplot as plt

# Define parameters
N = 100             # Number of samples
n = np.arange(N)    # Vector 0...N-1
f0 = 0.1            # Frequency
sigma = 1.5         # Amount of noise

# Create the sinusoid
x = np.sin(2 * np.pi * f0 * n)

# Add noise to the sinusoid
x += sigma * np.random.randn(x.size)

plt.plot(x, 'r-o')
