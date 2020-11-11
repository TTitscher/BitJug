#include "caster_petsc.h"
#include <pybind11/stl.h>
#include "MMA.h"

namespace py = pybind11;

void show(std::vector<Vec>& vs)
{
    for (auto& v : vs)
        VecView(v, PETSC_VIEWER_STDOUT_WORLD);
}

void timesTwo(Vec v)
{
    VecScale(v, 2.);
}

/**
 * The idea is to wrap the `class MMA` to
 *      - avoid modifications in the original code
 *      - provide verbose parameter names to ease the use
 *      - replace raw pointer lists with STL types for easy pybind11
 */
class PyMMA
{
public:
    PyMMA(PetscInt n_x, PetscInt n_constraints, Vec x)
        : mma(std::make_unique<MMA>(n_x, n_constraints, x))
    {
    }


    // Set and solve a subproblem: return new xval
    PetscErrorCode Update(Vec x_new, Vec df_dx, std::vector<PetscScalar> gx,
                          std::vector<Vec> dg_dx, Vec x_min, Vec x_max)
    {
        return mma->Update(x_new, df_dx, gx.data(), dg_dx.data(), x_min, x_max);
    }

    PetscErrorCode SetOuterMovelimit(PetscScalar Xmin, PetscScalar Xmax,
                                     PetscScalar movelim, Vec x, Vec xmin, Vec xmax)
    {
        return mma->SetOuterMovelimit(Xmin, Xmax, movelim, x, xmin, xmax);
    }


private:
    std::unique_ptr<MMA> mma;
};

PYBIND11_MODULE(py_mma, m)
{
    // toying around:
    m.doc() = "Hello PETSc pybind.";
    m.def("show", &show, "Show an existing PETSc vec.");
    m.def("times_two", &timesTwo, "Show an existing PETSc vec.");

    // actual MMA interface
    py::class_<PyMMA>(m, "MMA")
            .def(py::init<int, int, Vec>())
            .def("update", &PyMMA::Update)
            .def("set_outer_movelimit", &PyMMA::SetOuterMovelimit);
}

