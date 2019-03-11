#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone & Marco Baity-Jesi"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

import os
import logging
import subprocess
import numpy as np
import pandas as pd

from .aux import execution_parameters_permutations, current_datetime
from .aux import makedir_if_not_exist, order_of_magnitude

from . import logger  # noqa
lg = logging.getLogger(__name__)


TIME_ESTIMATOR_CONV = 2.5e-11  # hrs per particle per mc timestep per dimension
PROTOCOL_DICT = {
    1: "sample on the unit ball in a purely random fashion"
}
WARNING_MSG_LOCAL = """
Warning: you're attempting to run this on your local machine.
         This job could take days. Do you still wish to continue?

Proceed? (y for yes)
"""


def execute_protocol_1(params, target_run_directory, df, cluster=True,
                       prompt=True):
    """Docstring TODO"""

    executable = 'sbatch'

    if not cluster:
        executable = 'bash'

    process = subprocess.Popen("mv scripts/actual1.sbatch .",
                               shell=True, stdout=subprocess.PIPE)
    process.wait()

    exitcodes = []
    clean = True

    # Recall: `p` is a list of all permutations of the input parameters.
    for index, row in df.iterrows():
        beta = row['beta']
        dim = row['N']
        nmc = row['nmc']
        nvec = row['nvec']
        lmbdp = row['lambda_prime']
        ptype = row['ptype']
        n_report = params['n_report']
        d_save_energies = int(params['danger']['save_all_energies'])

        target_run_specific = os.path.join(target_run_directory, row['loc'])

        execution_string = \
            "%s actual1.sbatch 1 %f %i %i %i %f %s %s %i %i" \
            % (executable, beta, dim, nmc, nvec, lmbdp, ptype,
               target_run_specific, n_report, d_save_energies)

        process = subprocess.Popen(execution_string, shell=True,
                                   stdout=subprocess.PIPE,
                                   universal_newlines=True)
        process.wait()
        out, __ = process.communicate()
        exitcode = process.returncode
        exitcodes.append(exitcode)

        if exitcode != 0:
            clean = False

    process = subprocess.Popen("mv actual1.sbatch scripts/", shell=True,
                               stdout=subprocess.PIPE)
    process.wait()

    if clean:
        lg.info("Clean submission (protocol 1) ~ DONE")
    else:
        lg.error("Execution failed")
        lg.error("Error codes %a" % exitcodes)


def run_all(params, target_directory, prompt=True):
    """DOCSTRING: TODO"""

    p = execution_parameters_permutations(params['execution_parameters'])
    cluster = not params['danger']['run_on_local']
    protocol = params['protocol']
    Np = len(p)
    max_nmc = np.max(params['execution_parameters']['nmc'])
    max_nvec = np.max(params['execution_parameters']['nvec'])
    max_n = np.max(params['execution_parameters']['dims'])
    dt = current_datetime() + "-p%i" % protocol
    target_data_directory = os.path.join(target_directory, 'DATA_hdwell')
    target_run_directory = os.path.join(target_data_directory, dt)

    if prompt:
        print("\nProtocol %i (%s) ready to run"
              % (protocol, PROTOCOL_DICT[protocol]))
        print("    * Datetime directory %s will be created in" % dt)
        print("      %s" % target_data_directory)
        print("    * Total of %i independent jobs will be submitted run"
              % Np)
        print("    * Longest job runtime contains 10^%i monte carlo timesteps"
              % max_nmc)
        print("      and %i 'particles' (parallel executions)"
              % max_nvec)
        print("    * Estimated time to completion %.02f hours"
              % (TIME_ESTIMATOR_CONV * 10**max_nmc * max_nvec * max_n))
        print("    * Number of reports/job: %i" % params['n_report'])
        proceed = input("\nProceed? (y for yes)\n")

        if proceed != 'y':
            exit(0)

    # Log the parameter file, and create relevant directories.
    lg.info("%a" % params)
    lg.info("Creating %s if it does not already exist" % target_data_directory)
    makedir_if_not_exist(target_data_directory, error_out=False)
    lg.info("Creating %s (only if it doesn't exist)" % target_run_directory)
    makedir_if_not_exist(target_run_directory, error_out=True)

    # Within the target_run_directory, create all sub-directories. Each
    # corresponds to a different permutation of the parameters.
    zfill_index = order_of_magnitude(Np) + 1
    d = {
        'beta': [p_['beta'] for p_ in p],
        'N': [p_['dims'] for p_ in p],
        'nmc': [p_['nmc'] for p_ in p],
        'nvec': [p_['nvec'] for p_ in p],
        'lambda_prime': [params['lmbdp'] for iii in range(Np)],
        'ptype': [params['ptype'] for iii in range(Np)],
        'protocol': [params['protocol'] for iii in range(Np)],
        'loc': [str(iii).zfill(zfill_index) for iii in range(Np)]
    }
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(target_run_directory, 'params.csv'))

    # Make subdirectory for each permutation.
    for loc in d['loc']:
        path__ = os.path.join(target_run_directory, loc)
        makedir_if_not_exist(path__)

    lg.info("Proceeding to execute protocol %i" % protocol)
    if protocol == 1:
        execute_protocol_1(params, target_run_directory, df,
                           cluster=cluster, prompt=prompt)
