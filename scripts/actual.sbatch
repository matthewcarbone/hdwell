#!/bin/bash
#SBATCH --partition=New
#SBATCH --job-name=hdwell
#SBATCH --output=job_data/hdwell.%A.out
#SBATCH --error=job_data/hdwell.%A.err
#SBATCH --ntasks=1


# Variables:
#    01: TODO

python3 hdwell/sub.py ###"$1" "$2" "$3" "$4"
