project(py_mma LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.9)

list (APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}")

find_package(MPI REQUIRED)
find_package(PETSc REQUIRED)
find_package(pybind11 REQUIRED)
find_package(PETSc4py REQUIRED)


message(STATUS "PETSc INCLUDE DIRS: ${PETSC_INCLUDE_DIRS}")
message(STATUS "PETSc LIBS: ${PETSC_LIBRARIES}")


pybind11_add_module(py_mma py_mma.cpp MMA.cc)
#target_compile_options(py_mma PRIVATE -Wall;-Werror;-pedantic)

target_include_directories(py_mma PRIVATE ${PETSC4PY_INCLUDE_DIRS})
target_link_libraries(py_mma PRIVATE MPI::MPI_CXX PETSC::petsc pybind11::module)

