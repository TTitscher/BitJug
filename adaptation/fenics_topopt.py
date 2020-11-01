import numpy as np
import nlopt

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

    def goal(self, x, dobj=None):
        self.filt.filter_density(x)

        assemble(self.a, tensor=self.K)
        for bc in self.bcs:
            bc.apply(self.K, self.F)

        self.solver.solve(self.u.vector(), self.F)

        J = self.F.inner(self.u.vector())

        if dobj is not None:
            dc_dxf = assemble(self.dc_dxf_form)
            dobj[:] = self.filt.filter_sensitivity(dc_dxf)

        self.post_process()
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


class VolumeConstraint:
    def __init__(self, x, volfrac):
        self.x = x
        self.volfrac = volfrac
        self.V0 = assemble(Constant(1.0) * dx(x.function_space().mesh()))

        self.average = self.x / Constant(self.V0) * dx
        self.daverage = assemble(derivative(self.average, self.x))

    def __call__(self, x, daverage):
        self.x.vector()[:] = x
        daverage[:] = self.daverage[:]
        return assemble(self.average) - self.volfrac


class MeshFilter:
    def __init__(self, mesh, constraint=None):
        self.Vxf = FunctionSpace(mesh, "P", 1, constrained_domain=constraint)
        self.xf = Function(self.Vxf, name="xf")
        self.x = self.xf

    def filter_density(self, x=None):
        self.xf.vector()[:] = x

    def filter_sensitivity(self, dc_dxf):
        return dc_dxf[:]


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

        self.Lf = df * self.x * dx
        self.Tinv = assemble(TestFunction(self.Vxf) * TrialFunction(self.Vx) * dx)

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


def point_load_FEM(mesh, filt, prm, load_at):
    parameters["form_compiler"]["quadrature_degree"] = 1
    V = VectorFunctionSpace(mesh, "P", 1)
    F = Function(V).vector()
    P = PointSource(V.sub(1), Point(*load_at), 10)
    P.apply(F)

    bc = DirichletBC(V, np.zeros_like(load_at), plane_at(0.0, "x"))
    return FEM(V, filt, prm, F, [bc])


def define_optimizer(fem, volfrac, bounds=None):
    x0 = fem.filt.x.vector()[:]
    x0[:] = volfrac
    n = len(x0)

    opt = nlopt.opt(nlopt.LD_MMA, n)
    opt.set_min_objective(fem.goal)

    opt.add_inequality_constraint(VolumeConstraint(fem.filt.x, volfrac))

    if bounds is None:
        bounds = np.zeros(n), np.ones(n)

    opt.set_lower_bounds(bounds[0])
    opt.set_upper_bounds(bounds[1])

    return opt, x0


if __name__ == "__main__":
    lx, ly, lz = 60, 30, 20
    # lx, ly = 360, 120
    nx, ny, nz = lx, ly, lz
    load_at = (lx, ly / 2)

    from mshr import Box, generate_mesh

    mesh = RectangleMesh(Point(0, 0), Point(lx, ly), nx, ny)
    # mesh = BoxMesh(Point(0.,0.,0.), Point(lx, ly, lz), nx, ny, 4)
    # geometry = Box(Point(0,0,0), Point(lx, ly, lz))
    # mesh = generate_mesh(geometry, 120)

    # load_at = (lx, ly/2., lz/2.)

    filt = HelmholtzFilter(mesh, R=0.5)
    # filt = MeshFilter(mesh)
    fem = point_load_FEM(mesh, filt, Parameter(), load_at)

    opt, x0 = define_optimizer(fem, 0.1)
    opt.set_maxeval(50)
    x = opt.optimize(x0)
