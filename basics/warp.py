"""
Warp/Bend a line to a circle
============================
"""

import numpy as np
import matplotlib.pyplot as plt

"""
Interpret a 1D array of points as the circumference of a circle
centered at (0,0) and return the corresponding (x,y) coordinates.
"""

def warp(a):
    a0 = a - a[0]
    L = a0[-1]
    R = L / 2 / np.pi

    x = R * np.cos(a0 / R)
    y = R * np.sin(a0 / R)

    return x, y

if __name__ == "__main__":
    a = np.linspace(17.0, 42.0, 100)
    x, y = warp(a)
    def almost_equal(a, b):
        return abs(a-b) < 1.e-10
  
    # start point == end point
    assert almost_equal(x[0], x[-1]) and almost_equal(y[0], y[-1])

    # center (ignoring the duplicated end point) around (0,0)
    assert almost_equal(np.mean(x[:-1]), 0.) 
    assert almost_equal(np.mean(y[:-1]), 0.) 

    plt.plot(*warp(a))
    plt.show()
