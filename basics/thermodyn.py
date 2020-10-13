"""
Thermodynamic Topology Optimization
===================================

The implementation in [Jantos et al. 2019](https://dx.doi.org/10.1007/s00161-018-0706-y)
claims to be comparable in efficiency and resulting structrual compliance, but
* requires less (no?) artificial control parameters
* based purely on Newton-Raphson - no OC/MMA/SQP/...
* no artificial filters


Idea
----

"Inversion" of damage modeling


"""

from dolfin import *
from fenics_helpers.boundary import *
import numpy as np

E0 = 20000
nu = 0.2
def sigma(u):
    mu = E0 / (2.0 * (1.0 + nu))
    lmbda = E0 * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) 
    return 2.0 * mu * sym(grad(u)) + lmbda * tr(sym(grad(u))) * Identity(len(u))

def eps(u):
    return sym(grad(u))

lx, ly, nx, ny = 5, 1, 50, 10
mesh = RectangleMesh(Point(0,0), Point(lx, ly), nx, ny, "crossed")        

Vu = VectorFunctionSpace(mesh, "P", 1)
u, du, u_ = Function(Vu), TrialFunction(Vu), TestFunction(Vu)

Vxi = FunctionSpace(mesh, "DG", 0)
xi = Function(Vxi)

kappa = Constant(1e-5)
beta = Constant(1e-5)
rho = 1/(1 + (1/kappa - 1)*xi**3)

E = E0 * rho


u.vector()[:] = np.random.random(len(u.vector()[:]))
xi.vector()[:] += 0.4

p_m = 0.5 * inner(eps(u_), diff(rho,xi)* sigma(u)) * dx

print(assemble(p_m)[:])


# Psi_p =
