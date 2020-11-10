#include "caster_petsc.h"
#include "MMA.h"

namespace py = pybind11;

void show(Vec v)
{
    VecView(v, PETSC_VIEWER_STDOUT_WORLD);
}

PYBIND11_MODULE(py_mma, m)
{
    // toying around:
    m.doc() = "Hello PETSc pybind.";
    m.def("show", &show, "Show an existing PETSc vec.");

    // actual MMA interface
    // py::class_<MMA>(m, "MMA")
    //        .def(py::init<int, int, Vec>(), "Construct using defaults subproblem
    //        penalization")
    //        //.def("update", &MMA::Update, "Set and solve a subproblem: return new
    //        xval"); .def("restart", &MMA::Restart, "Return necessary data for possible
    //        restart") .def("set_asymptotes", &MMA::SetAsymptotes, "Set the aggresivity
    //        of the moving asymptotes");
}

