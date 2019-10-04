#!/bin/bash
#SBATCH --partition=New
#SBATCH --job-name=hdwell
#SBATCH --output=job_data/hdwell.%A.out
#SBATCH --error=job_data/hdwell.%A.err
#SBATCH --ntasks=1
#SBATCH --exclude=compute-0-[0-2,6]

# Variables:
#    01: N_MC
#    02: N_dims
#    03: N_tracer
#    04: beta
#    05: delta
#    06: ptype
#    07: protocol
#    08: lambdaprime
#    09: dw
#    10: save to where

export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1

python3 sub.py "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}"
