"""
Warp/Bend a line to a circle
============================
"""

import numpy as np
import matplotlib.pyplot as plt

def warp(a):
    a0 = a-a[0]
    L = a0[-1]
    R = L / 2 / np.pi
    
    x = R * np.cos(2 * np.pi * a0/L)
    y = R * np.sin(2 * np.pi * a0/L)

    return x, y

a = np.linspace(17., 42., 100)


plt.plot(*warp(a))
plt.show()

