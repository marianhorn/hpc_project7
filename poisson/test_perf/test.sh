#!/bin/bash

# Test the runtime performance of:
# - PETSc: using using CG solve " -ksp_type cg "
# - sp_dir: Scipy Direct Sparse Solver (uses COLAMD)
# - dn_dir: Numpy dense solver using LAPACK routine _gesv.
# - sp_cg: Scipy CG

LOG_FILE=log.log
> $LOG_FILE 

# TODO: Nothing, just run the script.

NUMP_PROC=1
for SIZE in 8 16 32 64 128; do
    echo "" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
    echo "Running PETSc solver for ${SIZE}x${SIZE}" | tee -a $LOG_FILE
    mpirun -np $NUMP_PROC ../poisson_petsc -ksp_type cg -nx $SIZE -ny $SIZE 2>&1 | tee -a $LOG_FILE
    echo "Running Python solver for ${SIZE}x${SIZE}" | tee -a $LOG_FILE
    python3 ../poisson_py.py -nx $SIZE -ny $SIZE 2>&1 | tee -a $LOG_FILE
done

NUMP_PROC=1
for SIZE in 256 512; do
    echo "" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
    echo "Running PETSc solver for ${SIZE}x${SIZE}" | tee -a $LOG_FILE
    mpirun -np $NUMP_PROC ../poisson_petsc -ksp_type cg -nx $SIZE -ny $SIZE 2>&1 | tee -a $LOG_FILE
    echo "Running Python solver for ${SIZE}x${SIZE}" | tee -a $LOG_FILE
    python3 ../poisson_py.py -nx $SIZE -ny $SIZE -solver="sp_dir,sp_cg" 2>&1 | tee -a $LOG_FILE
done