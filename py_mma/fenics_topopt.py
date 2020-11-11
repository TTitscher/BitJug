import numpy as np
# import nlopt

from dolfin import *
from fenics_helpers.boundary import *


class Parameter:
    def __init__(self):
        self.simp_exponent_p = 3.0
        self.simp_eps = 1.0e-6

        self.youngs_modulus = 20000.0
        self.poissons_ratio = 0.2

        self.solver = "direct"

        self.xdmf_file_name = "out.xdmf"


def simp(x, prm):
    p = Constant(prm.simp_exponent_p)
    _eps = Constant(prm.simp_eps)
    return _eps + (1 - _eps) * x ** p


def eps(u):
    return sym(grad(u))


def sigma(u, prm):
    E, nu = Constant(prm.youngs_modulus), Constant(prm.poissons_ratio)
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return 2.0 * mu * sym(grad(u)) + lmbda * tr(sym(grad(u))) * Identity(len(u))

def x_at(coordinate=0, eps=1.0e-10):
    class B(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], coordinate, eps)
    return B()


class FEM:
    def __init__(self, V, filt, prm, F, bcs):
        self.filt = filt
        self.prm = prm

        self.u, du, u_ = Function(V), TrialFunction(V), TestFunction(V)
        self.a = simp(filt.xf, prm) * inner(eps(u_), sigma(du, prm)) * dx

        self.pp = XDMFFile(prm.xdmf_file_name)
        self.pp.parameters["functions_share_mesh"] = True
        self.pp.parameters["flush_output"] = True
        self.t = 0.0

        self.dKdxU = derivative(self.a * self.u, filt.xf)
        self.dc_dxf_form = -derivative((self.a * self.u) * self.u, filt.xf)
        self.dxf = Function(filt.Vxf)

        self.F = F
        self.bcs = bcs

        self.K = PETScMatrix()

        if prm.solver == "iterative":
            self.null_space = build_nullspace(V, self.u.vector())
            self.K.set_near_nullspace(self.null_space)

            pc = PETScPreconditioner("petsc_amg")

            # Use Chebyshev smoothing for multigrid
            PETScOptions.set("mg_levels_ksp_type", "chebyshev")
            PETScOptions.set("mg_levels_pc_type", "jacobi")

            # Improve estimate of eigenvalues for Chebyshev smoothing
            PETScOptions.set("mg_levels_esteig_ksp_type", "gmres")
            PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)

            self.solver = PETScKrylovSolver("gmres", pc)
            self.solver.set_operator(self.K)
        else:
            self.solver = LUSolver(self.K, "mumps")

    def goal(self, x, dobj):
        self.filt.filter_density(x)

        assemble(self.a, tensor=self.K)
        for bc in self.bcs:
            bc.apply(self.K, self.F)

        self.solver.solve(self.u.vector(), self.F)
        J = self.F.inner(self.u.vector())

        dc_dxf = PETScVector()
        assemble(self.dc_dxf_form, tensor=dc_dxf)
        dc_dxf.vec().ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        v = self.filt.filter_sensitivity(dc_dxf).vec()
        assign(v, dobj)

        self.post_process()
        if PETSc.COMM_WORLD.rank == 0:
            print(f"{int(self.t):3d} c = {J:10.6f}", flush=True)
        return J

    def post_process(self):
        """
        Called once per "goal" evaluation 
        """
        self.pp.write(self.filt.x, self.t)
        self.pp.write(self.filt.xf, self.t)
        self.t += 1.0


def build_nullspace(V, x):
    """Function to build null space for 3D elasticity"""

    # Create list of vectors for null space
    nullspace_basis = [x.copy() for i in range(6)]

    # Build translational null space basis
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0)
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0)
    V.sub(2).dofmap().set(nullspace_basis[2], 1.0)

    # Build rotational null space basis
    V.sub(0).set_x(nullspace_basis[3], -1.0, 1)
    V.sub(1).set_x(nullspace_basis[3], 1.0, 0)
    V.sub(0).set_x(nullspace_basis[4], 1.0, 2)
    V.sub(2).set_x(nullspace_basis[4], -1.0, 0)
    V.sub(2).set_x(nullspace_basis[5], 1.0, 1)
    V.sub(1).set_x(nullspace_basis[5], -1.0, 2)

    for x in nullspace_basis:
        x.apply("insert")

    # Create vector space basis and orthogonalize
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    return basis

from petsc4py import PETSc
def assign(source, dest):
    dest.setArray(source.getArray())
    dest.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

class VolumeConstraint:
    def __init__(self, x, volfrac):
        self.x = x
        self.volfrac = volfrac
        self.V0 = assemble(Constant(1.0) * dx(x.function_space().mesh()))

        self.average = self.x / Constant(self.V0) * dx

        self.daverage = PETScVector()
        assemble(derivative(self.average, self.x), tensor=self.daverage)

    def __call__(self, x, daverage):
        assign(x, self.x.vector().vec())
        assign(self.daverage.vec(), daverage)
        return assemble(self.average) - self.volfrac


class MeshFilter:
    def __init__(self, mesh, constraint=None):
        self.Vxf = FunctionSpace(mesh, "P", 1, constrained_domain=constraint)
        self.xf = Function(self.Vxf, name="xf")
        self.x = self.xf

    def filter_density(self, x=None):
        assign(x, self.x.vector().vec())

    def filter_sensitivity(self, dc_dxf):
        return dc_dxf


class HelmholtzFilter:
    def __init__(self, mesh, R, constraint=None):
        self.R = Constant(R)
        self.Vx = FunctionSpace(mesh, "DG", 0, constrained_domain=constraint)
        self.Vxf = FunctionSpace(mesh, "P", 1, constrained_domain=constraint)

        self.x = Function(self.Vx, name="x")
        self.xf = Function(self.Vxf, name="xf")

        df, f_ = TestFunction(self.Vxf), TrialFunction(self.Vxf)
        af = dot(grad(df), R ** 2 * grad(f_)) * dx + df * f_ * dx
        self.Kf = assemble(af)
        self.solver = KrylovSolver(self.Kf, "cg", "jacobi")
        # self.solver = LUSolver(self.Kf)

        self.Lf = df * self.x * dx
        self.Tinv = assemble(df * TrialFunction(self.Vx) * dx)

    def filter_density(self, x=None):
        if x is not None:
            # self.x.vector().set_local(x.getArray())
            # self.x.vector().apply("insert")
            assign(x, self.x.vector().vec())
        self.solver.solve(self.xf.vector(), assemble(self.Lf))

    def filter_sensitivity(self, dc_dxf):
        # 1) apply the filter which will transform d
        Ldc_form = TestFunction(self.Vxf) * Function(self.Vxf, dc_dxf) * dx
        Ldc = PETScVector()
        assemble(Ldc_form, tensor=Ldc)
        dc_dx_in_xf = PETScVector()
        
        Ldc.apply("insert")

        self.solver.solve(dc_dx_in_xf, Ldc)
        dc_dx_in_xf.apply("insert")

        # transform
        dc_dx = PETScVector()
        self.Tinv.transpmult(dc_dx_in_xf, dc_dx)
        dc_dx.apply("insert")
        return dc_dx


def point_load_FEM(mesh, filt, prm, load_at):
    V = VectorFunctionSpace(mesh, "P", 1)
    F = Function(V).vector()
    P = PointSource(V.sub(1), Point(*load_at), 10)
    P.apply(F)

    bc = DirichletBC(V, np.zeros_like(load_at), x_at(0.0))
    return FEM(V, filt, prm, F, [bc])


if __name__ == "__main__":

    import py_mma
    
    parameters["form_compiler"]["quadrature_degree"] = 1

    F = 1
    lx, ly= F*120, F*60
    nx, ny= lx, ly
    load_at = (lx, ly / 2)

    mesh = RectangleMesh(Point(0, 0), Point(lx, ly), nx, ny)

    filt = HelmholtzFilter(mesh, R=1.)
    filt = MeshFilter(mesh)
    fem = point_load_FEM(mesh, filt, Parameter(), load_at)


    x0 = fem.filt.x.vector().vec().copy()
    x0.set(0.4)
    Xmin, Xmax = 0., 1.

    xmin = x0.copy()
    xmin.set(Xmin)
    
    xmax = x0.copy()
    xmax.set(Xmax)

    dc = x0.copy()
    dg = x0.copy()


    for v in [x0, xmin, xmax]:
        v.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    vv = VolumeConstraint(fem.filt.x, 0.2)
   
    mma = py_mma.MMA(x0.getSize(), 1, x0)
    print(x0.getSize())

    for i in range(50):
        c = fem.goal(x0, dc)
        g = vv(x0, dg)
        
        # exit(0)

        mma.set_outer_movelimit(Xmin, Xmax, 0.2, x0, xmin, xmax)
        mma.update(x0, dc, [g], [dg], xmin, xmax)

        assign(x0, fem.filt.x.vector().vec())
        XDMFFile("stuffy.xdmf").write(fem.filt.x)
    # print(vv(x0, dc))



    # opt, x0 = define_optimizer(fem, 0.4)
    # opt.set_maxeval(50)
    # x = opt.optimize(x0)
