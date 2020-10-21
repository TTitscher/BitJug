import nlopt
import numpy as np

def objective(x, dobj):
    dobj[:] = 1
    return x[0]

def constraint(x, dconstr):
    dconstr[:] = 1
    return x[0] + 0.5



n = 1
x0 = np.ones(n)
opt = nlopt.opt(nlopt.LD_MMA,n)

opt.set_min_objective(objective)
opt.add_inequality_constraint(constraint)

opt.set_upper_bounds([1])
opt.set_lower_bounds([0])

opt.set_maxeval(1000)
opt.set_xtol_rel(1.e-4)
print(opt.optimize(x0))
print(opt.get_numevals())

