# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 15:09:30 2016

@author: hehu
"""

import matplotlib.pyplot as plt
import numpy as np

plt.close("all")

N = 100
n = np.arange(N) # Vector [0,1,2,...,N-1]
x = np.cos(2 * np.pi * n * 0.03)
x_noisy = x + 0.2 * np.random.randn(N)

fig = plt.figure(figsize = [10,5])

plt.plot(n, x, 'r-', linewidth = 2, label = "Clean Sinusoid")
plt.plot(n, x_noisy, 'bo-', markerfacecolor = "green", label = "Noisy Sinusoid")

plt.grid("on")
plt.xlabel("Time in $\mu$s")
plt.ylabel("Amplitude")
plt.title("An Example Plot")
plt.legend(loc = "upper left")

plt.show()
plt.savefig("../images/sinusoid.pdf", 
            bbox_inches = "tight", transparent = "True")
