from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.sparse import lil_matrix, coo_matrix

mesh = UnitSquareMesh(20,20, "crossed")

V = FunctionSpace(mesh, "P", 1)

r = V.tabulate_dof_coordinates()

R = 0.2
A = lil_matrix((len(r), len(r)))

for dof0, x0 in enumerate(tqdm.tqdm(r)):
    for dof1, x1 in enumerate(r):
        dist = np.linalg.norm(x0 - x1)
        if dist < R:
            A[dof0, dof1] = R - dist


plt.spy(A)
plt.show()

    # print(dof, ":", x)
