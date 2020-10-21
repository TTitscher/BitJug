from dolfin import *
import numpy as np
import mpi4py

"""
Collect all entries of a GenericVector/PETScVector on rank 0 in a parallel run.
Why?
   Combine a (compute intensive, thus parallel) fenics simulation with a
   comparatively fast (but only available in serial) optimization algorithm.
"""
def rank():
    return MPI.rank(MPI.comm_world)

def assign_from_0(numpy_v0, v):
    if rank() != 0:
        numpy_v0 = np.empty(v.size(), dtype="float64")

    MPI.comm_world.Bcast([numpy_v0, mpi4py.MPI.DOUBLE], root=0)

    r0, r1 = v.local_range()
    v.zero()
    v.add_local(numpy_v0[r0:r1])
    v.apply("insert")

def main():
    mesh = UnitSquareMesh(100,100)
    V = FunctionSpace(mesh, "P", 1)
    u = interpolate(Expression("sin(x[0])", degree=2), V).vector()
    
    u_new = Function(V).vector()

    u_on_0 = u.gather_on_zero()
    # print(u_on_0)
    # print(u_on_0)
    assign_from_0(u_on_0, u_new)
    u_new *= 0.7

    u_new_on_0 = u_new.gather_on_zero()
    if rank() == 0:
        print(np.max(np.abs(u_new_on_0 - u_on_0*0.7)))

if __name__ == "__main__":
    main()
    
# from mpi4py import MPI
# # import numpy as np
#
# comm = MPI.COMM_WORLD
# nproc = comm.Get_size()
# rank = comm.Get_rank()
#
#
# scal = None
# mat = np.empty([3,3], dtype='d')
#
# arr = np.empty(5, dtype='d')
# result = np.empty(5, dtype='d')
#
#
# if rank==0:
#     scal = 55.0
#     mat[:] = np.array([[1,2,3],[4,5,6],[7,8,9]])
#
#     arr = np.ones(5)
#     result = 2*arr
#
# comm.Bcast([ result , MPI.DOUBLE], root=0)
# scal = comm.bcast(scal, root=0)
# comm.Bcast([ mat , MPI.DOUBLE], root=0)
#
# print("Rank: ", rank, ". Array is:\n", result)
# print("Rank: ", rank, ". Scalar is:\n", scal)
# print("Rank: ", rank, ". Matrix is:\n", mat)
