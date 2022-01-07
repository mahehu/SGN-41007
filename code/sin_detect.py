# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:38:19 2018

@author: hehu
"""

import numpy as np

def detect(x, frequency, Fs, L = 128):

    n = np.arange(L)
    h = np.exp(-2 * np.pi * 1j * frequency * n / Fs)
    y = np.abs(np.convolve(h, x, 'same'))
    
    return y
    