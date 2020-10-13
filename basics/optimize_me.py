from scipy.optimize import *
import numpy as np


def principle(N=100):
    f = lambda x:np.linalg.norm(x)
    cc = {"type": "ineq", "fun": lambda x:np.sum(x) - N + x[0]}

    result = minimize(f, np.ones(N)*6174, constraints=cc, method="SLSQP")

    print(result)

    print(cc["fun"](result.x))


principle()
