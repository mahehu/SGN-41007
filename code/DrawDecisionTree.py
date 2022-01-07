# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:14:19 2015

@author: hehu
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

if __name__ == "__main__":

    ax = plt.subplot(111)
    

    plt.figure(figsize=(3,2))
    ax=plt.axes([0.1, 0.1, 0.8, 0.7])
    an1 = ax.annotate("Test 1", xy=(0.5, 0.5), xycoords="data",
                      va="center", ha="center",
                      bbox=dict(boxstyle="round", fc="w"))
    
    from matplotlib.text import OffsetFrom
    offset_from = OffsetFrom(an1, (0.5, 0))
    an2 = ax.annotate("Test 2", xy=(0.1, 0.1), xycoords="data",
                      xytext=(0, -10), textcoords=offset_from,
                      # xytext is offset points from "xy=(0.5, 0), xycoords=an1"
                      va="top", ha="center",
                      bbox=dict(boxstyle="round", fc="w"),
                      arrowprops=dict(arrowstyle="->"))