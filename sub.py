#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone & Marco Baity-Jesi"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"


import sys
from hdwell.aux import pure_mc_sampling


def protocol1(beta, dim, nmc, nvec, lambda_prime, ptype, report_energy,
              report_psi_basin, report_psi_config, report_memory,
              target_run_directory, n_report):

    obs = {
        'report_energy': bool(report_energy),
        'report_psi_basin': bool(report_psi_basin),
        'report_psi_config': bool(report_psi_config),
        'report_memory': bool(report_memory)
    }
    pure_mc_sampling(dim, beta, lambda_prime, nmc, nvec, ptype, obs, n_report,
                     target_run_directory)


if __name__ == '__main__':
    protocol = int(sys.argv[1])
    beta = float(sys.argv[2])
    dim = int(sys.argv[3])
    nmc = int(sys.argv[4])
    nvec = int(sys.argv[5])
    lambda_prime = float(sys.argv[6])
    ptype = str(sys.argv[7])
    report_energy = int(sys.argv[8])
    report_psi_basin = int(sys.argv[9])
    report_psi_config = int(sys.argv[10])
    report_memory = int(sys.argv[11])
    target_run_directory = str(sys.argv[12])
    n_report = int(sys.argv[13])

    if protocol == 1:
        protocol1(beta, dim, nmc, nvec, lambda_prime, ptype, report_energy,
                  report_psi_basin, report_psi_config, report_memory,
                  target_run_directory, n_report)
