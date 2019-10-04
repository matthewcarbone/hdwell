#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"


import sys
import pickle

from hdwell2.simulator import simulate

if __name__ == '__main__':

    # Execute the simulation (this should be run directly on a cluster)
    N_mc_LOG = int(sys.argv[1])
    N_dims = int(sys.argv[2])
    N_tracer = int(sys.argv[3])
    beta = float(sys.argv[4])
    delta = float(sys.argv[5])
    ptype = str(sys.argv[6])
    protocol = int(sys.argv[7])
    lambdaprime = float(sys.argv[8])
    dw = float(sys.argv[9])
    save_to = str(sys.argv[10])

    print(sys.argv)

    all_results = simulate(
        N_mc_LOG, N_dims, N_tracer, beta, delta, ptype, protocol,
        save_all_energies=False,
        n_report=1000,
        verbose=True,
        lambdaprime=lambdaprime,
        dw=dw,
        n_e_sample=1000,
        n_mem_sample=100)

    # Next save the results to disk:
    HP = pickle.HIGHEST_PROTOCOL
    pickle.dump(all_results, open(save_to, 'wb'), protocol=HP)
