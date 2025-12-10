#!/bin/bash

# This script tests if the code runs correctly. The norm of the solution must be the same!

# TODO: Nothing, just run the script.

# Run the Python solver
python3 ../poisson_py.py -nx 128 -ny 32

# Visualize the results for dense and sparse solvers
python3 ../draw.py solution_dn_dir
python3 ../draw.py solution_sp_dir

# Run the PETSc solver
mpirun -np 1 ../poisson_petsc -nx 128 -ny 32

# Visualize the PETSc solution
python3 ../draw.py solution_petsc
