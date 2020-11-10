// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <petsc4py/petsc4py.h>
#include <petscvec.h>
#include <pybind11/pybind11.h>

// pybind11 casters for PETSc/petsc4py objects

// Import petsc4py on demand
#define VERIFY_PETSC4PY(func)                                                            \
    if (!func)                                                                           \
    {                                                                                    \
        if (import_petsc4py() != 0)                                                      \
            throw std::runtime_error("Error when importing petsc4py");                   \
    }

// Macro for casting between PETSc and petsc4py objects

namespace pybind11
{
namespace detail
{
template <>
class type_caster<_p_Vec>
{
public:
    PYBIND11_TYPE_CASTER(Vec, _("vec"));
    bool load(handle src, bool)
    {
        if (src.is_none())
        {
            value = nullptr;
            return true;
        }
        VERIFY_PETSC4PY(PyPetscVec_Get);
        if (PyObject_TypeCheck(src.ptr(), &PyPetscVec_Type) == 0)
            return false;
        value = PyPetscVec_Get(src.ptr());
        return true;
    }

    static handle cast(Vec src, pybind11::return_value_policy policy, handle parent)
    {
        VERIFY_PETSC4PY(PyPetscVec_New);
        auto obj = PyPetscVec_New(src);
        if (policy == pybind11::return_value_policy::take_ownership)
            PetscObjectDereference((PetscObject)src);
        return pybind11::handle(obj);
    }

    operator Vec()
    {
        return value;
    }
};
} // namespace detail
} // namespace pybind11

