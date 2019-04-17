#!/bin/bash
#SBATCH --partition=New
#SBATCH --job-name=mcmc_rf
#SBATCH --output=job_data/mcmc_rf.%A.out
#SBATCH --error=job_data/mcmc_rf.%A.err
#SBATCH --ntasks=1
#SBATCH -N 1

python3 rfinder.py