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


def run_all(params, target_directory, prompt=True):
    """DOCSTRING: TODO"""

    p = execution_parameters_permutations(params['execution_parameters'])
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
        print("    * Total of %i independent jobs will be submitted via SLURM"
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
        pass
