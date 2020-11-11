import numpy as np
from petsc4py import PETSc
import py_mma
import nlopt


N = 10000


def reference():
    def f(x, df):
        df[:] = 2 * x
        return np.linalg.norm(x) **2

    def g(x, dg):
        dg[:] = -1.
        return 42. - sum(x)

    opt = nlopt.opt(nlopt.LD_MMA, N)
    opt.set_min_objective(f)

    opt.set_lower_bounds(np.zeros(N))
    opt.set_upper_bounds(np.ones(N) * 100)
    
    opt.add_inequality_constraint(g)

    x0 = np.ones(N)

    opt.set_maxeval(100)
    x = opt.optimize(x0)

    print(x)


def with_mma():
    def v(x):
        return PETSc.Vec().createWithArray(np.array(x), len(x))

    x = v(np.ones(N))
    x_new = v(np.ones(N))
    Xmin = 0.0
    Xmax = 100.0
    xmin = v(np.ones(N) * Xmin)
    xmax = v(np.ones(N) * Xmax)
    mma = py_mma.MMA(N, 1, x)

    def f(x):
        return x.norm() **2

    def df(x):
        return 2 * x

    def g(x):
        return 42. - x.sum()

    def dg(x):
        return v(-np.ones(N) )

    for i in range(100):

        v_df = df(x)

        movelimit = 0.1
        mma.set_outer_movelimit(Xmin, Xmax, movelimit, x, xmin, xmax)

        gg, dgg = g(x), dg(x)
        mma.update(x_new, v_df, [gg], [dgg], xmin, xmax)

        ff = f(x)
        dx = (x_new - x).norm()
        if PETSc.COMM_WORLD.rank == 0:
            print(i, ff, dx)
        # x[:] = x_new[:]
        x.setArray(x_new.getArray())
        # x_new.copy(x)

        if dx < 1.0e-8:
            break

    print(x.getArray())

# @profile
def main():
    reference()
    with_mma()

main()
