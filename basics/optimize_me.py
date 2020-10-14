from scipy.optimize import *
import numpy as np
import nlopt

def principle(N=100, shut_it=True):
    f = lambda x:np.linalg.norm(x)
    cc = {"type": "ineq", "fun": lambda x:np.sum(x) - N + x[0]}

    result = minimize(f, np.ones(N)*6174, constraints=cc, method="SLSQP")

    if shut_it:
        return
    print(result)
    print(cc["fun"](result.x))


principle()

from dolfin import *
from fenics_helpers.boundary import *

## SIMP
def simp(x):
    p = Constant(3.)
    _eps = Constant(1.e-6)
    return _eps + (1 - _eps) * x ** p

class FEM:
    def __init__(self, lx, ly, nx, ny, E0=20000, nu=0.2):
        self.V0 = lx * ly

        self.mesh = RectangleMesh(Point(0,0), Point(lx, ly), nx, ny, "crossed")        
        self.Vu = VectorFunctionSpace(self.mesh, "P", 1)
        self.force = Constant((0, -1))

        self.Vx = FunctionSpace(self.mesh, "DG", 0)
        self.x = interpolate(Constant(0.4),self.Vx)

        self.vol = self.x * dx
        self.bcs = []
        self.bcs.append(DirichletBC(self.Vu, (0.,0.), plane_at(0., "x")))
        
        def eps(u):
            return sym(grad(u))

        def sigma(u):
            mu = E0 / (2.0 * (1.0 + nu))
            lmbda = E0 * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) 
            return 2.0 * mu * sym(grad(u)) + lmbda * tr(sym(grad(u))) * Identity(len(u))

        du, u_ = TrialFunction(self.Vu), TestFunction(self.Vu)
        F = simp(self.x)*inner(eps(u_), sigma(du)) * dx + inner(self.force, u_)*dx
        self.a, self.L = lhs(F), rhs(F)
        self.u = Function(self.Vu)

    def volume_constraint(self, x, dv=None):
        dv[:] = 1.
        self.x.vector()[:] = x
        return -assemble(self.vol) /self.V0 + 0.4
    

    def goal(self,x, dobj=None):
        print("now", flush=True)
        self.x.vector()[:] = x

        solve(self.a == self.L, self.u, self.bcs)
       
        J = dot(self.force, self.u) + Constant(10.0) * dot(grad(self.x), grad(self.x))
        return assemble(J * dx)
        
        




if __name__ == "__main__":
    set_log_level(50)
    fem = FEM(20, 5, 20, 5)

    x0 = fem.x.vector()[:]
    n = len(x0)

    opt = nlopt.opt(nlopt.LD_MMA, n)
    opt.set_lower_bounds(np.zeros(n))
    opt.set_upper_bounds(np.ones(n))

    opt.set_min_objective(fem.goal) 
    opt.add_inequality_constraint(fem.volume_constraint) 


    ff = XDMFFile("yay.xdmf")
    ff.parameters["functions_share_mesh"]=True
    ff.parameters["flush_output"]=True
    t = 0.
    def callback(x,_):
        global t
        print("Yay", t)
        fem.x.vector()[:]=x
        ff.write(fem.x, t)
        t += 1.

    x = opt.optimize(x0)
    print(result)
        


    # print(fem.volume_constraint(x0))
    # print(fem.goal(x0))
    # print(fem.goal(x0*2))

    

