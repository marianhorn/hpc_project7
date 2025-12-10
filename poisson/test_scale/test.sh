#!/bin/bash

# Strong Scaling for performance of PETSc with " -ksp_type cg" :

# TODO: Nothing, just run the script.

LOG_FILE=log.log
> $LOG_FILE

SIZE=1024 
for NUMP_PROC in 1 2 4 8 16; do
    echo "####################################################" | tee -a $LOG_FILE
    echo "Running PETSc solver with $NUMP_PROC processes for grid size ${SIZE}x${SIZE}" | tee -a $LOG_FILE
    echo "####################################################" | tee -a $LOG_FILE
    mpirun -np $NUMP_PROC ../poisson_petsc -ksp_type cg -nx $SIZE -ny $SIZE -log_view :performance_log_$NUMP_PROC.log 2>&1 | tee -a $LOG_FILE
done
