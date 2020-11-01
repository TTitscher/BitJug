from fenics_topopt import *
from mshr import Box, generate_mesh

lx, ly, lz = 60, 30, 20

nx, ny, nz = lx, ly, lz
load_at = (lx, ly, lz)


geometry = Box(Point(0,0,0), Point(lx, ly, lz))
mesh = generate_mesh(geometry, 50)

prm  = Parameter()
prm.solver = "iterative"
prm.xdmf_file_name = "topopt3d.xdmf"

filt = HelmholtzFilter(mesh, R=0.5)
fem = point_load_FEM(mesh, filt, prm, load_at)

opt, x0 = define_optimizer(fem, 0.1)
print("Optimization problem of dim", len(x0))
opt.set_maxeval(50)
x = opt.optimize(x0)
