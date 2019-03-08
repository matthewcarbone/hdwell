#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone & Marco Baity-Jesi"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

import os
import logging
import subprocess
import numpy as np

from .aux import execution_parameters_permutations, current_datetime
from .aux import makedir_if_not_exist

from . import logger  # noqa
lg = logging.getLogger(__name__)


TIME_ESTIMATOR_CONV = 2.5e-9  # hrs per particle per mc timestep
PROTOCOL_DICT = {
    1: "sample on the unit ball in a purely random fashion"
}
WARNING_MSG_LOCAL = """
Warning: you're attempting to run this on your local machine.
         This job could take days. Do you still wish to continue?

Proceed? (y for yes)
"""


def execute_protocol_1(params, p, target_run_directory, cluster=True,
                       prompt=True):
    executable = 'sbatch'

    if not cluster and prompt:
        ans = input(WARNING_MSG_LOCAL)
        if ans == 'y':
            executable = 'bash'
        else:
            return

    process = subprocess.Popen("mv scripts/actual1.sbatch .",
                               shell=True, stdout=subprocess.PIPE)
    process.wait()

    exitcodes = []
    clean = True

    # Recall: `p` is a list of all permutations of the input parameters.
    for p_ in p:
        beta = p_['beta']
        dim = p_['dims']
        nmc = p_['nmc']
        nvec = p_['nvec']
        lmbdp = params['lmbdp']
        ptype = params['ptype']
        report_energy = int(params['observables']['average_energy'])
        report_psi_basin = int(params['observables']['psi_basin'])
        report_psi_config = int(params['observables']['psi_config'])
        report_memory = int(params['observables']['memory'])
        n_report = params['n_report']
        execution_string = \
            "%s actual1.sbatch 1 %f %i %i %i %f %s %i %i %i %i %s %i" \
            % (executable, beta, dim, nmc, nvec, lmbdp, ptype,
               report_energy, report_psi_basin, report_psi_config,
               report_memory, target_run_directory, n_report)
        process = subprocess.Popen(execution_string, shell=True,
                                   stdout=subprocess.PIPE,
                                   universal_newlines=True)
        process.wait()
        out, __ = process.communicate()
        print(out)
        exitcode = process.returncode
        exitcodes.append(exitcode)
        break

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
    cluster = not params['run_on_local']
    protocol = params['protocol']
    Np = len(p)
    max_nmc = np.max(params['execution_parameters']['nmc'])
    max_nvec = np.max(params['execution_parameters']['nvec'])
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
              % (TIME_ESTIMATOR_CONV * 10**max_nmc * max_nvec))
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

    # Make a copy of the yaml file and save it in the target_run_directory.
    lg.info("Copying YAML parameter file to this run's data directory")
    process = subprocess.Popen("cp %s/params.yaml %s"
                               % (target_directory, target_run_directory),
                               shell=True, stdout=subprocess.PIPE)
    process.wait()

    lg.info("Proceeding to execute protocol %i" % protocol)
    if protocol == 1:
        execute_protocol_1(params, p, target_run_directory,
                           cluster=cluster, prompt=prompt)
