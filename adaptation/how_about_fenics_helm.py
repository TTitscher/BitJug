import numpy as np
import nlopt

from dolfin import *
from fenics_helpers.boundary import *

parameters["form_compiler"]["quadrature_degree"]=1

def simp(x):
    p = Constant(3.)
    _eps = Constant(1.e-6)
    return _eps + (1 - _eps) * x ** p
        
    
class MeshFilter:
    def __init__(self, mesh, **kwargs):
        self.Vxf = FunctionSpace(mesh, "P", 1)
        self.xf = Function(self.Vxf, name="xf")
        self.x = self.xf

        self.V0 = Constant(assemble(Constant(1.)*dx(mesh)))
        self.average = self.xf/self.V0 * dx
        self.daverage = assemble(derivative(self.average, self.xf))

    def filter_density(self, x=None):
        self.xf.vector()[:] = x
    
    def filter_sensitivity(self, dc_dxf):
        return dc_dxf[:]

    def average_density(self, x, daverage):
        self.xf.vector()[:] = x
        daverage[:] = self.daverage[:]
        return assemble(self.average)

class HelmholtzFilter:
    def __init__(self, mesh, R):
        self.R = Constant(R)
        self.Vx = FunctionSpace(mesh, "DG", 0)
        self.Vxf = FunctionSpace(mesh, "P", 1)
        
        self.x = Function(self.Vx, name="x")
        self.xf = Function(self.Vxf, name="xf")
    
        df, f_ = TestFunction(self.Vxf), TrialFunction(self.Vxf)
        af = dot(grad(df), R**2 * grad(f_)) * dx + df * f_ * dx
        self.Kf = assemble(af)
        self.solver = KrylovSolver(self.Kf, "cg", "jacobi")    

        self.Lf = df * self.x * dx 
        self.Tinv = assemble(TestFunction(self.Vxf) * TrialFunction(self.Vx)*dx)

        self.V0 = Constant(assemble(Constant(1.)*dx(mesh)))
        self.average = self.x/self.V0 * dx
        self.daverage = assemble(derivative(self.average, self.x))

    def filter_density(self, x=None):
        if x is not None:
            self.x.vector()[:] = x
        self.solver.solve(self.xf.vector(), assemble(self.Lf))

    def filter_sensitivity(self, dc_dxf):
        # 1) apply the filter which will transform d
        Ldc_form = TestFunction(self.Vxf) * Function(self.Vxf, dc_dxf) * dx
        Ldc = assemble(Ldc_form)
        dc_dx_in_xf = Function(self.Vxf)
        
        self.solver.solve(dc_dx_in_xf.vector(), Ldc)

        # transform
        dc_dx = Function(self.Vx)
        self.Tinv.transpmult(dc_dx_in_xf.vector(), dc_dx.vector())
        return dc_dx.vector()[:]

    def average_density(self, x, daverage):
        self.x.vector()[:] = x
        daverage[:] = self.daverage[:]
        return assemble(self.average)
        

class FEM:
    def __init__(self, mesh, load_at, filt, E0=20000, nu=0.2):

        self.filt = filt
        self.Vu = VectorFunctionSpace(mesh, "P", 1)

        self.dxf = Function(filt.Vxf)

        self.bcs = []
        self.bcs.append(DirichletBC(self.Vu, (0.,0.), plane_at(0., "x")))
       
        
        def eps(u):
            return sym(grad(u))

        def sigma(u):
            mu = E0 / (2.0 * (1.0 + nu))
            lmbda = E0 * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) 
            return 2.0 * mu * sym(grad(u)) + lmbda * tr(sym(grad(u))) * Identity(len(u))

        du, u_ = TrialFunction(self.Vu), TestFunction(self.Vu)
        self.a = simp(filt.xf)*inner(eps(u_), sigma(du)) * dx #ä+ inner(self.force, u_)*ds(1)
        self.u = Function(self.Vu)

        self.dK = diff(simp(filt.xf), filt.xf)*inner(eps(u_), sigma(du)) * dx 
    
        self.pp = XDMFFile("yay.xdmf")
        self.pp.parameters["functions_share_mesh"]=True
        self.pp.parameters["flush_output"]=True
        self.t = 0.


        self.F = Function(self.Vu).vector()
        P = PointSource(self.Vu.sub(1), Point(*load_at), 10)
        P.apply(self.F)
        
        
        self.dKdxU = derivative(self.a * self.u, filt.xf)
        self.dc_dxf_form = -derivative((self.a * self.u) * self.u, filt.xf)

        self.K = PETScMatrix()
        self.solver = LUSolver(self.K, "mumps")

    def goal(self, x, dobj=None):
        self.filt.filter_density(x)
       
        assemble(self.a, tensor=self.K)
        for bc in self.bcs:
            bc.apply(self.K)

        self.solver.solve(self.u.vector(), self.F)

        J = self.F.inner(self.u.vector())
    
        if dobj is not None:
            dc_dxf = assemble(self.dc_dxf_form)
            dobj[:] = self.filt.filter_sensitivity(dc_dxf)

        self.pp.write(self.filt.x, self.t)
        self.pp.write(self.filt.xf, self.t)
        self.t += 1.

        print(f"{int(self.t):3d} c = {J:10.6f}", flush=True)
        return J
        
if __name__ == "__main__":
    set_log_level(50)
   
    lx, ly = 180, 60
    # lx, ly = 360, 120
    nx, ny = lx, ly
    load_at = (lx, ly/2)
    mesh = RectangleMesh(Point(0,0), Point(lx, ly), nx, ny)        

    filt = HelmholtzFilter(mesh, R=1)
    filt = MeshFilter(mesh)
    fem = FEM(mesh, load_at, filt)

    volfrac = 0.3
    volume_constraint = lambda x, diff: filt.average_density(x, diff) - volfrac

    x0 = filt.x.vector()[:]
    x0[:] = volfrac
    n = len(x0)
    print(n)
  
    cdf = False
    if cdf:
        dd = np.zeros(n)
        fem.goal(x0, dd)
        
        delta = 1.e-4
        dd_cdf = np.empty(n)
        for i in range(n):
            deltax = np.zeros_like(x0)
            deltax[i] = delta

            dd_cdf[i] = (fem.goal(x0 + deltax) - fem.goal(x0 - deltax))/(2*delta)
           
        print(dd / dd_cdf)
        # print(dd_cdf)
        exit(-1)




    opt = nlopt.opt(nlopt.LD_MMA, n)
    opt.set_lower_bounds(np.zeros(n))
    opt.set_upper_bounds(np.ones(n))

    opt.set_min_objective(fem.goal) 
    opt.add_inequality_constraint(volume_constraint) 

    opt.set_maxeval(50)
    x = opt.optimize(x0)
