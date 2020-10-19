import numpy as np
import nlopt
import matplotlib.pyplot as plt

from dolfin import *
from fenics_helpers.boundary import *

from scipy.sparse import lil_matrix, csr_matrix
import tqdm

## SIMP
def simp(x):
    p = Constant(3.)
    # return x**p
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
        self.xf = interpolate(Constant(0.0),self.Vx)
        self.dxf = interpolate(Constant(0.0),self.Vx)

        self.vol = self.x/self.V0 * dx
        self.bcs = []
        self.bcs.append(DirichletBC(self.Vu, (0.,0.), plane_at(0., "x")))
        
        def load(x, on_boundary):
            return near(x[1], ly) and on_boundary 

        facets = MeshFunction("size_t", self.mesh, 1)
        AutoSubDomain(load).mark(facets, 1)
        ds = Measure("ds", subdomain_data=facets)

        def eps(u):
            return sym(grad(u))

        def sigma(u):
            mu = E0 / (2.0 * (1.0 + nu))
            lmbda = E0 * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) 
            return 2.0 * mu * sym(grad(u)) + lmbda * tr(sym(grad(u))) * Identity(len(u))

        du, u_ = TrialFunction(self.Vu), TestFunction(self.Vu)
        F = simp(self.xf)*inner(eps(u_), sigma(du)) * dx + inner(self.force, u_)*ds(1)
        self.a, self.L = lhs(F), rhs(F)
        self.u = Function(self.Vu)

        self.dK = diff(simp(self.xf), self.xf)*inner(eps(u_), sigma(du)) * dx 
    
        self.pp = XDMFFile("yay.xdmf")
        self.pp.parameters["functions_share_mesh"]=True
        self.pp.parameters["flush_output"]=True
        self.t = 0.

        #
        self.R = 0.05
        # r = self.Vx.tabulate_dof_coordinates()
        # A = lil_matrix((len(r), len(r)))
        #
        # for dof0, x0 in enumerate(tqdm.tqdm(r)):
        #     for dof1, x1 in enumerate(r):
        #         dist = np.linalg.norm(x0 - x1)
        #         if dist < R:
        #             A[dof0, dof1] = R - dist
    


    

    def volume_constraint(self, x, dv=None):
        self.x.vector()[:] = x
        dv[:] = assemble(derivative(self.vol, self.x))[:]
        return assemble(self.vol) - 0.4


    def goal(self,x, dobj=None):
        self.x.vector()[:] = x
        VDG = FunctionSpace(self.mesh, "DG", 0)
        VP = FunctionSpace(self.mesh, "P", 1)
        df, f_ = TestFunction(VP), TrialFunction(VP)

        af = dot(grad(df), self.R**2 * grad(f_)) * dx + df * f_ * dx
        Lf = df * self.x * dx 
        
        A = assemble(af)
        L = assemble(Lf)
        
        solve(A, self.xf.vector(), L)

        Vx = assemble(self.x * dx)
        Vxf = assemble(self.xf * dx)
        print(Vx - Vxf)
        
        K, F = assemble_system(self.a, self.L, self.bcs)
        solve(K, self.u.vector(), F)

        J = F.inner(self.u.vector())
        dKdxU = assemble(derivative(self.a * self.u, self.xf))
        print(self.t, "c = ", J, flush=True)
    
        # help(dKdxU.transpmult)
        # exit(-1)
        dc_dx = Function(self.Vx)
        dKdxU.transpmult(self.u.vector(), dc_dx.vector())

        Ldx = assemble(-df * dc_dx * dx)
        
        solve(A, self.dxf.vector(), Ldx)

        dobj[:] = self.dxf.vector()[:]
        
        self.pp.write(self.x, self.t)
        self.pp.write(self.xf, self.t)
        self.t += 1.

        return J
        
        




if __name__ == "__main__":
    set_log_level(50)
    fem = FEM(4, 1, 200, 50)

    x0 = fem.x.vector()[:]
    n = len(x0)
    
    dd = np.zeros(n)
    fem.goal(x0, dd)
    # print(dd)
    
    # exit(-1)

    opt = nlopt.opt(nlopt.LD_MMA, n)
    opt.set_lower_bounds(np.zeros(n))
    opt.set_upper_bounds(np.ones(n))

    opt.set_min_objective(fem.goal) 
    opt.add_inequality_constraint(fem.volume_constraint) 



    opt.set_maxeval(100)
    # opt.set_xtol_abs(0.1)
    x = opt.optimize(x0)


    print(x)
        


    # print(fem.volume_constraint(x0))
    # print(fem.goal(x0))
    # print(fem.goal(x0*2))

    

