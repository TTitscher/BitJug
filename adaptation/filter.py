from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.sparse import lil_matrix, coo_matrix

mesh = UnitSquareMesh(20,20, "crossed")

V = FunctionSpace(mesh, "P", 1)

def density_alogrithm(V, u, R=0.2):
    r = V.tabulate_dof_coordinates()

    A = lil_matrix((len(r), len(r)))

    for dof0, x0 in enumerate(tqdm.tqdm(r)):
        for dof1, x1 in enumerate(r):
            dist = np.linalg.norm(x0 - x1)
            if dist < R:
                A[dof0, dof1] = R - dist

    f = Function(V)
    f.vector()[:] = A * u.vector()[:]
    return f

def density_helmholtz(V, u, R=0.2):
    df, f_ = TestFunction(V), TrialFunction(V)
    f = Function(V)
        
    af = dot(grad(df), R**2 * grad(f_)) * dx + df * f_ * dx
    Lf = df * u * dx 

    solve(af==Lf, f)

    return f

u = Function(V)
u.vector()[100] = 1.

f = density_helmholtz(V,u)
print(f.vector()[:])
plot(f)
plt.show()

