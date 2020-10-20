import numpy as np
import nlopt

from dolfin import *
from fenics_helpers.boundary import *

parameters["form_compiler"]["quadrature_degree"]=1

def simp(x):
    p = Constant(4.)
    _eps = Constant(1.e-6)
    return _eps + (1 - _eps) * x ** p

class FEM:
    def __init__(self, lx, ly, nx, ny, E0=20000, nu=0.2):
        self.V0 = lx * ly

        self.mesh = RectangleMesh(Point(0,0), Point(lx, ly), nx, ny, "crossed")        
        self.Vu = VectorFunctionSpace(self.mesh, "P", 1)
        self.force = Constant((0, -1))

        self.Vx = FunctionSpace(self.mesh, "DG", 0)
        self.Vxf = FunctionSpace(self.mesh, "P", 1)

        self.x = Function(self.Vx, name="x")
        self.x.vector()[:] = 0.2

        self.xf = Function(self.Vxf, name="xf")
        self.dxf = Function(self.Vxf)

        self.vol = self.x/self.V0 * dx
        self.bcs = []
        self.bcs.append(DirichletBC(self.Vu, (0.,0.), plane_at(0., "x")))
       
        
        def eps(u):
            return sym(grad(u))

        def sigma(u):
            mu = E0 / (2.0 * (1.0 + nu))
            lmbda = E0 * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) 
            return 2.0 * mu * sym(grad(u)) + lmbda * tr(sym(grad(u))) * Identity(len(u))

        du, u_ = TrialFunction(self.Vu), TestFunction(self.Vu)
        self.a = simp(self.xf)*inner(eps(u_), sigma(du)) * dx #ä+ inner(self.force, u_)*ds(1)
        self.u = Function(self.Vu)

        self.dK = diff(simp(self.xf), self.xf)*inner(eps(u_), sigma(du)) * dx 
    
        self.pp = XDMFFile("yay.xdmf")
        self.pp.parameters["functions_share_mesh"]=True
        self.pp.parameters["flush_output"]=True
        self.t = 0.

        self.R = lx/nx
    
        df, f_ = TestFunction(self.Vxf), TrialFunction(self.Vxf)

        af = dot(grad(df), self.R**2 * grad(f_)) * dx + df * f_ * dx
        self.Lf = df * self.x * dx 
        
        self.Kf = assemble(af)
        self.filterer = LUSolver(self.Kf)
        
        self.T = assemble(TestFunction(self.Vxf) * TrialFunction(self.Vx)*dx)
        
        self.F = Function(self.Vu).vector()
        P = PointSource(self.Vu.sub(1), Point(lx, ly/2.), 10)
        P.apply(self.F)
        
        self.dv = assemble(derivative(self.vol, self.x))[:]
        
        self.dKdxU = derivative(self.a * self.u, self.xf)
        
        self.dc_dxf = Function(self.Vxf)
        self.Ldx = -TestFunction(self.Vxf) * self.dc_dxf * dx

        self.K = PETScMatrix()

    def volume_constraint(self, x, dv=None):
        self.x.vector()[:] = x
        dv[:] = self.dv
        return assemble(self.vol) - 0.2

    def goal(self,x, dobj=None):
        self.x.vector()[:] = x
        
        self.filterer.solve(self.xf.vector(), assemble(self.Lf))

        assert abs(assemble(self.x * dx) - assemble(self.xf * dx)) < 1.e-10
      
        assemble(self.a, tensor=self.K)
        for bc in self.bcs:
            bc.apply(self.K)

        solve(self.K, self.u.vector(), self.F)

        J = self.F.inner(self.u.vector())
        print(self.t, "c = ", J, flush=True)
    
        dKdxU = assemble(self.dKdxU)
        dKdxU.transpmult(self.u.vector(), self.dc_dxf.vector())

        Ldx = assemble(self.Ldx)
        
        self.filterer.solve(self.dxf.vector(), Ldx)

        Tdxf = Function(self.Vx)

        self.T.transpmult(self.dxf.vector(), Tdxf.vector())

        dobj[:] = Tdxf.vector()[:]
        
        self.pp.write(self.x, self.t)
        self.pp.write(self.xf, self.t)
        self.t += 1.

        return J
        
if __name__ == "__main__":
    set_log_level(50)
    fem = FEM(200, 50, 200, 50)

    x0 = fem.x.vector()[:]
    n = len(x0)
    
    dd = np.zeros(n)
    fem.goal(x0, dd)

    opt = nlopt.opt(nlopt.LD_MMA, n)
    opt.set_lower_bounds(np.zeros(n))
    opt.set_upper_bounds(np.ones(n))

    opt.set_min_objective(fem.goal) 
    opt.add_inequality_constraint(fem.volume_constraint) 

    opt.set_maxeval(50)
    x = opt.optimize(x0)
