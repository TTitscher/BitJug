"""
Topology optimization
=====================

... following the Matlab implementation of [2014 Liu&Tovar](https://doi.org/10.1007/s00158-014-1107-x).

Mesh
----

"""
from dolfin import *
from fenics_helpers.boundary import *

class FEM:
    def __init__(self, lx, ly, nx, ny, E0=20000, nu=0.2):
        self.mesh = RectangleMesh(Point(0,0), Point(lx, ly), nx, ny, "crossed")        
        self.Vu = VectorFunctionSpace(self.mesh, "P", 2)
        du, u_ = TrialFunction(self.Vu), TestFunction(self.Vu)


        self.Vx = FunctionSpace(self.mesh, "P", 2)
        self.x = interpolate(Constant(1.),self.Vx)

        self.p = Constant(3.)
        self.Emin = Constant(E0 * 1.e-5)

        self.E = self.Emin + self.x**self.p * (E0 - self.Emin) # SIMP

        def eps(u):
            return sym(grad(u))

        def sigma(u):
            mu = self.E / (2.0 * (1.0 + nu))
            lmbda = self.E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) 
            return 2.0 * mu * sym(grad(u)) + lmbda * tr(sym(grad(u))) * Identity(len(u))

        self.F = inner(eps(u_), sigma(du)) * dx + inner(Constant((0,0)),u_)*dx

if __name__ == "__main__":
    fem = FEM(20, 5, 200, 50)
    bc0 = DirichletBC(fem.Vu, (0.,0.), plane_at(0., "x"))
    bc1 = DirichletBC(fem.Vu.sub(0), 42., plane_at(20., "x"))
    A, b = assemble_system(lhs(fem.F), rhs(fem.F), [bc0, bc1])
    u = Function(fem.Vu)
    solve(A, u.vector(), b)
    XDMFFile("fem.xdmf").write(u)
