# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:25:09 2018

@author: hehu
"""

# Example of playing a noisy sinusoid
# Requires installation of sounddevice module.
# In anaconda: "pip install sounddevice" should do this.

import sounddevice as sd
import numpy as np
from sin_detect import detect
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    f = 1000    # Play a beep at this Hz
    Fs = 8192   # Samplerate
    sigma = 1   # Stddev of the noise
    
    n = np.arange(3 * Fs) # The sample times
    x = np.sin(2 * np.pi * f * n / Fs)
    
    # Zero out the beginning and the end
    x[:Fs] = 0
    x[2*Fs:] = 0
    
    # Add noise
    x_noisy = x + sigma * np.random.randn(*x.shape)
    
    # Play the sound
    sd.play(x_noisy, Fs)
    
    # Detect
    y = detect(x_noisy, frequency = f, Fs = Fs)
    
    # Plot results
    
    fig, ax = plt.subplots(3, 1)
    
    ax[0].plot(n, x)
    ax[0].set_title("Ground truth sinusoid")
    ax[0].grid("on")
    
    ax[1].plot(n, x_noisy)
    ax[1].set_title("Noisy sinusoid")
    ax[1].grid("on")
    
    ax[2].plot(n, y)
    ax[2].set_title("Detection result")
    ax[2].grid("on")
    
    plt.tight_layout()
    
    plt.show()
    