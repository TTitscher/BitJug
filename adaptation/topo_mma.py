import numpy as np
from topopt.boundary_conditions import MBBBeamBoundaryConditions
from topopt.problems import ComplianceProblem
from topopt.filters import DensityBasedFilter
from topopt.guis import GUI
import nlopt

nelx, nely = 120, 40  # Number of elements in the x and y
volfrac = 0.4  # Volume fraction for constraints
penal = 3.0  # Penalty for SIMP
rmin = 1.4  # Filter radius

# Initial solution
x = volfrac * np.ones(nely * nelx, dtype=float)

# Boundary conditions defining the loads and fixed points
bc = MBBBeamBoundaryConditions(nelx, nely)

# Problem to optimize given objective and constraints
problem = ComplianceProblem(bc, penal)
gui = GUI(problem, "Topology Optimization Example")
topopt_filter = DensityBasedFilter(nelx, nely, rmin)

def volume_constraint(x, dv):
    topopt_filter.filter_variables(x, xPhys)
    dv[:] = 1. 
    topopt_filter.filter_volume_sensitivities(xPhys, dv)
    return xPhys.sum() - volfrac * x.size


def objective(x, dc):
    topopt_filter.filter_variables(x, xPhys)
    obj = problem.compute_objective(xPhys, dc)
    topopt_filter.filter_objective_sensitivities(xPhys, dc)
    gui.update(xPhys)
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




