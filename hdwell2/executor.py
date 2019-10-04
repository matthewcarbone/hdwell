#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

"""Generates the input files and executes the main scripts. Should import only
modules from auxiliary."""

import os
import yaml
import pickle
import subprocess as sp

import hdwell2.auxiliary as aux

WORKDIR = os.getcwd()


def load_parameters():
    """Loads and returns the paramter file p.yaml in the working directory."""

    return yaml.safe_load(open(os.path.join(WORKDIR, 'p.yaml')))


def generate_directory_info(p, makedirs=True):
    """Returns a string containing the date and time of the run, as well as the
    protocol. Also returns the absolute path to the data directory for the
    run. If makedirs is set to True, it will also make the directories of the
    absolute path."""

    dt = aux.current_datetime() + "-p%i" % p['protocol']
    abs_path = os.path.join(WORKDIR, 'DATA_hdwell', dt)

    if makedirs:
        os.makedirs(abs_path)

    return dt, abs_path


def _get_job_string(ii, jj, ii_zfill, jj_zfill, protocol):
    """Returns the job string used in `generate_job_indexes`."""

    _s_ii = str(ii).zfill(ii_zfill)
    _s_jj = str(jj).zfill(jj_zfill)

    return "p%s-%s-%s" % (str(protocol), _s_ii, _s_jj)


def generate_job_indexes(p):
    """Creates an index for every job. Specifically, the index will look like
    this: pX-XXXXX-XXXXX, where pX indexes the protocol, the first set of
    numbers indexes the parameter combination, and the second indexes simply a
    duplicate run using the same parameters (used in averaging later). Returns
    a dictionary in which each key is this string, and the value is the
    dictionary of parameters for that run."""

    combos = aux.execution_parameters_permutations(p['combo'])
    n_combos_zfill = aux.order_of_magnitude(len(combos)) + 1
    protocol = p['protocol']
    mult_zfill = aux.order_of_magnitude(p['mult']) + 1

    d_final = {}

    # Iterate over parameter combinations
    for ii, dict in enumerate(combos):

        # Iterate over identical runs
        for jj in range(p['mult']):

            job_string = _get_job_string(
                ii, jj, n_combos_zfill, mult_zfill, protocol)

            d_final[job_string] = dict

    return d_final


def prepare(p, d_final):
    """Gets ready for the run by 1) creating all necessary directories and
    2) parsing and saving all of the parameter combinations. Returns the
    abs_path for use in dispatch."""

    dt, abs_path = generate_directory_info(p, makedirs=True)

    # Now save the dictionary as a pickle file
    _loc = os.path.join(abs_path, "d_all.pkl")
    pickle.dump(d_final, open(_loc, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    return abs_path


def dispatch():
    """Sends off the jobs to the job controller, in this case SLURM."""

    p = load_parameters()
    d_final = generate_job_indexes(p)
    abs_path = prepare(p, d_final)

    total_jobs = len(d_final)
    ui = input("Total jobs to submit: %i, continue? [y = yes]\t" % total_jobs)

    if ui != 'y':
        raise RuntimeError("User exited.")

    # Iterate over all jobs, first Move the submit script to the working
    # directory
    process = sp.Popen(
        "mv scripts/submit.sbatch .", shell=True, stdout=sp.PIPE)
    process.wait()

    for key, value in d_final.items():
        save_to = os.path.join(abs_path, key + ".pkl")
        str_ex = \
            "sbatch submit.sbatch %i %i %i %f %f %s %i %f %f %s" \
            % (value['N_MC'], value['N_dims'], value['N_tracer'],
               value['beta'], value['delta'], p['ptype'], p['protocol'],
               p['lambdaprime'], value['dw'], save_to)

        process = sp.Popen(str_ex, shell=True, stdout=sp.PIPE,
                           universal_newlines=True)
        process.wait()
        exitcode = process.returncode
        if exitcode != 0:
            print("Execution error %a" % exitcode)

    # Move submit script back to its home
    process = sp.Popen(
        "mv submit.sbatch scripts/", shell=True, stdout=sp.PIPE)
    process.wait()

    print("Loop finished.")
