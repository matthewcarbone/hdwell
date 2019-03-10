#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone & Marco Baity-Jesi"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"


import sys
from hdwell.aux import pure_mc_sampling


def protocol1(beta, dim, nmc, nvec, lambda_prime, ptype,
              target_run_directory, n_report, save_all_energies):

    pure_mc_sampling(dim, beta, lambda_prime, nmc, nvec, ptype, n_report,
                     target_run_directory, save_all_energies)


if __name__ == '__main__':
    protocol = int(sys.argv[1])
    beta = float(sys.argv[2])
    dim = int(sys.argv[3])
    nmc = int(sys.argv[4])
    nvec = int(sys.argv[5])
    lambda_prime = float(sys.argv[6])
    ptype = str(sys.argv[7])
    target_run_directory = str(sys.argv[8])
    n_report = int(sys.argv[9])
    save_all_energies = bool(int(sys.argv[10]))

    if protocol == 1:
        protocol1(beta, dim, nmc, nvec, lambda_prime, ptype,
                  target_run_directory, n_report, save_all_energies)
