#!/bin/bash
#SBATCH --partition=All
#SBATCH --job-name=hdwell
#SBATCH --output=job_data/hdwell.%A.out
#SBATCH --error=job_data/hdwell.%A.err
#SBATCH --ntasks=1


# Variables:
#    01: Protocol number (1)
#    02: Beta (inverse temperature 1/T)
#    03: Number of dimensions N
#    04: Number of monte carlo time steps
#    05: Length of the vector (number of "particles" in the "ensemble")
#    06: Lambdaprime
#    07: Potential type
#    08: Target run directory to save observables
#    09: Number of times to report
#    10: Save all energies?
#    11: Save all statistical information?
#    12: Markov chain?
#    13: Exponential distribution scale parameter

python3 sub.py "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" \
"${12}" "${13}"
