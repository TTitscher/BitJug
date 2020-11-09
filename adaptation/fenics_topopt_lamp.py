from fenics_topopt import *
from fenics_helpers.boundary import *
from mshr import *



prm  = Parameter()
prm.xdmf_file_name = "lamp2d.xdmf"


lx, ly = 100, 20
geometry = Rectangle(Point(0,0), Point(lx, ly))
mesh = generate_mesh(geometry, 100)

class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)

    def map(self, x, y):
        y[0] = x[0] - lx
        y[1] = x[1] 


V = VectorFunctionSpace(mesh, "P", 1)
parameters["form_compiler"]["quadrature_degree"] = 1

bc = DirichletBC(V, Constant((0,0)), plane_at(0, "y"))

facets = MeshFunction("size_t", mesh, 1)
load_boundary = plane_at(ly, "y")
load_boundary.mark(facets, 42)
ds = Measure("ds", subdomain_data=facets)

p = Constant((1., -5.))

F = assemble(dot(TestFunction(V), p) * ds(42) + dot(TestFunction(V), Constant((0, -10)))*dx)

filt = HelmholtzFilter(mesh, R=0.2, constraint=PeriodicBoundary())

fem = FEM(V, filt, prm, F, [bc])

opt, x0 = define_optimizer(fem, 0.5)
print("Optimization problem of dim", len(x0))
opt.set_maxeval(200)
x = opt.optimize(x0)
