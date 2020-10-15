import numpy as np
import math
import nlopt
import scipy

from topopt.boundary_conditions import *
from topopt.problems import ComplianceProblem
from topopt.guis import GUI

# nelx, nely = 120, 30  # Number of elements in the x and y
nelx, nely = 360, 200  # Number of elements in the x and y
volfrac = 0.4  # Volume fraction for constraints
penal = 3.0  # Penalty for SIMP
rmin = 5  # Filter radius

show_gui = True

# Initial solution
x = volfrac * np.ones(nely * nelx, dtype=float)

# Boundary conditions defining the loads and fixed points
bc = CantileverBoundaryConditions(nelx, nely)

# Problem to optimize given objective and constraints
problem = ComplianceProblem(bc, penal)

def H_filter(nelx, nely, rmin):
        """
        Create a filter to filter solutions.
        Build (and assemble) the index+data vectors for the coo matrix format.
        nelx:
            The number of elements in the x direction.
        nely:
            The number of elements in the y direction.
        rmin:
            The filter radius.
        """
        nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1)**2))
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        cc = 0
        for i in range(nelx):
            for j in range(nely):
                row = i * nely + j
                kk1 = int(max(i - (math.ceil(rmin) - 1), 0))
                kk2 = int(min(i + math.ceil(rmin), nelx))
                ll1 = int(max(j - (math.ceil(rmin) - 1), 0))
                ll2 = int(min(j + math.ceil(rmin), nely))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * nely + l
                        fac = rmin - math.sqrt(
                            ((i - k) * (i - k) + (j - l) * (j - l)))
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = max(0.0, fac)
                        cc = cc + 1
        # Finalize assembly and convert to csc format
        H = scipy.sparse.coo_matrix(
            (sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
        Hs = H.sum(axis=1)
        return H, Hs

H, Hs = H_filter(nelx, nely, rmin)


if show_gui:
    gui = GUI(problem, "Topology Optimization Example")

def volume_constraint(x, dv):
    xPhys[:] = np.asarray(H * x[:,np.newaxis] / Hs)[:, 0]
    dv[:] = 1.
    return xPhys.sum() - volfrac * x.size


def objective(x, dc):
    xPhys[:] = np.asarray(H * x[:,np.newaxis] / Hs)[:, 0]
    obj = problem.compute_objective(xPhys, dc)
    dc[:] = np.asarray(H * (dc[:,np.newaxis] / Hs))[:, 0]

    if show_gui:
        gui.update(xPhys)
    print(obj, flush=True)
    return obj

def optimize():
    xPhys = x.copy()
    return solver.opt.optimize(x)

n = nelx * nely
opt = nlopt.opt(nlopt.LD_MMA, n)

xPhys = np.ones(n)

opt.set_lower_bounds(np.zeros(n))
opt.set_upper_bounds(np.ones(n))

opt.set_min_objective(objective)
opt.add_inequality_constraint(volume_constraint, 0)

opt.set_maxeval(30)

opt.optimize(x)



