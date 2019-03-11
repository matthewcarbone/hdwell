#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone & Marco Baity-Jesi"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

"""All auxiliary/support functions."""

import numpy as np
import os
import sys
from math import floor, log2, log10
import logging
import datetime
import pickle
from time import time
from itertools import product
from collections import Counter
import secrets

from . import logger  # noqa
lg = logging.getLogger(__name__)


AVAILABLE_P_TYPES = ['quad', 'log']
ENERGY_SAMPLE = 1000


# Directory & Data Tools ------------------------------------------------------
def makedir_if_not_exist(path, error_out=False):
    """Does exactly what is says it does: if the directory specified by `path`
    does not exist, create it."""

    if not os.path.isdir(path):
        lg.debug("%s does not exist. Creating." % path)
        os.mkdir(path)
    else:
        if error_out:
            raise FileExistsError("%s exists already, will not overwrite."
                                  % path)
        lg.debug("%s exists." % path)


def current_datetime():
    """Get's the current date and time and converts it into a string to use
    for tagging files."""

    now = datetime.datetime.now()
    dt = now.strftime("%Y-%m-%d-%H-%M-%S")
    lg.info("Datetime logged as %s" % dt)
    return dt


def get_random_hash(n=16):
    return secrets.token_hex(n)

# -----------------------------------------------------------------------------


def order_of_magnitude(x):
    """Returns the base 10 order of magnitude of `x`."""

    return int(log10(x))


def energy(x, lmbd=1.0, ptype='log'):
    """The potential energy as a function of some input `x`.

    Parameters
    ----------
    x : array_like
        Usually the positions of an ensemble of particles. The dimension-axis
        should always be axis 1.
    ptype : str
        Available potential energies. Default is 'log'. Available options:
            'quad' : Simple harmonic oscillator form v = lmbd*x*x.
            'log'  : Logarithmic form (natural log) v = lmbd^{-1}*ln(x)
    lmbd : float
        Multiplier parameter. Default is 1.0.
    """

    if ptype not in AVAILABLE_P_TYPES:
        raise RuntimeError("Fatal: invalid potential surface type `ptype`.")

    if ptype == 'quad':
        return lmbd * np.sum(x**2)
    elif ptype == 'log':
        return np.log(np.sqrt(np.sum(x**2, axis=1))) / lmbd


def sample_nball(N, nMC=1, n_vec=100):
    """Uses the `smart` sampling method as described by Barthe, Guedon,
    Mendelson and Naor in Annals of Probability 33(2), 480 (2005). Samples on
    the unit N-ball uniformally.

    Parameters
    ----------
    N : int
        Dimension of the system.
    nMC : int
        Number of monte carlo time steps, should always be 1. Default is 1.
    n_vec : int
        Number of "particles" in the "ensemble". Default is 100.

    Returns a NumPy array which contains positions for every dimension,
    monte carlo timestep and particle. Again note that because the update
    scheme depends on prior positions nMC=1 is really a necessity.
    """

    x_vector = np.random.normal(scale=1.0 / np.sqrt(2.0), size=(nMC, N, n_vec))
    sum_of_squares = np.sum(x_vector**2, axis=1, keepdims=True)
    y = np.random.exponential(scale=1.0, size=(nMC, 1, n_vec))
    vec_final = x_vector / np.sqrt(sum_of_squares + y)
    return vec_final


def execution_parameters_permutations(dictionary):
    """Inputs a dictionary of a format such as

    eg = {
        hp1: [1, 2]
        hp2: [3, 4]
    }

    and returns a list of all permutations:

    eg1 = {
        hp1: 1
        hp2: 3
    }

    eg2 = {
        hp1: 1
        hp2: 4
    }

    eg3 = {
        hp1: 2
        hp2: 3
    }

    eg4 = {
        hp1: 2
        hp2: 4
    }
    """

    combinations = [dict(zip(dictionary, prod))
                    for prod in product(*(dictionary[ii]
                                          for ii in dictionary))]
    lg.info("Combinations sampled: total of %i permutations"
            % len(combinations))
    return combinations


def thresholds(N, beta, lambdaprime, ptype='log'):
    """Returns the threshold radius and energy. Note `lambdaprime` is equal
    to the critical inverse temperature beta_c."""

    if beta == 1:
        r = np.exp(-1.0 / N)
    else:
        r = (2.0 - beta / lambdaprime)**(1.0 / (N * beta / lambdaprime - N))

    if ptype == 'log':
        e = N * np.log(r) / lambdaprime
    else:
        raise RuntimeError("ptype %s not supported." % ptype)

    return [r, e]


def pure_mc_sampling(N, beta, lambdaprime, nMC_lg, n_vec, ptype, n_report,
                     data_directory, save_all_energies, verbose=True):
    """Executes a purely random sampling algorithm over the unit N-ball.
    Note that this function is meant to be called from within a compute node.
    It is assumed that all output will be piped to a SLURM (or related) output
    file and therefore no logging will be used here.

    Parameters
    ----------
    N : int
        Dimension of the system.
    beta : float
        Inverse temperature 1/T.
    lambdaprime : float
        Scale parameter set as lambdaprime = lambda * N. Generally, the
        potential will be defined something like lambda^{-1}*ln x, and this
        lambdaprime is fixed such that the DOS of our system matches that of
        literature precedent. Ultimately, lambdaprime=1 is a good default and
        probably should not be changed. This scaling also guarantees an
        extensive energy with respect to the dimension.
    nMC_lg : int
        Total number of monte carlo timesteps to perform is 10^nMC_lg
    n_vec : int
        Total number of "particles" in the "ensemble".
    ptype : str
        Determines the energy of the system (fixes the form of the potential
        energy).
    n_report : int
        Report progress (pipe to output) everytime the simulation reaches this
        many timesteps.
    data_directory : str
        Absolute path to the location of the data directory to which
        observables will be saved.
    save_all_energies : bool
        If true, will save the 10^nMC-dimensional vector containing the average
        energies at every timestep.
    """

    nMC = int(10**nMC_lg)
    lambd_ = lambdaprime / N
    increment = int(nMC / n_report)  # TODO: if increment is 0
    zfill_index = order_of_magnitude(nMC) + 1
    [r_threshold, e_threshold] = thresholds(N, beta, lambdaprime, ptype=ptype)

    x0 = sample_nball(N, nMC=1, n_vec=n_vec)     # Get the initial position
    e0 = energy(x0, lmbd=lambd_, ptype=ptype)    # Initial energy
    t0 = time()                                  # Initial time

    # Initialize empty dictionaries/arrays for each possible observable
    # (except the memory, done later). For the dictionaries, the key will be
    # the order of magnitude corresponding to the length of the time in some
    # basin/configuration, and the value will be the number of entires in the
    # histogram.
    avg_e = []
    psi_basin = Counter({})
    psi_config = Counter({})

    # Initialize the number of timesteps in the current basin / config. Each
    # time a "particle" changes basins / configuration, this counter will
    # reset.
    n_basin = np.zeros((n_vec))
    n_config = np.ones((n_vec))

    # Some particles will already be in a basin.
    n_basin[np.where(e0 < e_threshold)[1]] += 1

    # Timesteps at which to sample the energy.
    sample_e = np.logspace(0, nMC_lg, ENERGY_SAMPLE, dtype=int, endpoint=True)

    # Initialize the energy vector if report save all energies is true.
    if save_all_energies:
        all_e = np.empty((nMC, n_vec))

    # Temporarily ignore overflow warnings, since exp(large number) = inf and
    # this is fine for the code.
    np.seterr(over='ignore')

    # Begin the MC process.
    for ii in range(nMC + 1):

        # Report at designated timesteps.
        if ii % increment == 0 and verbose:
            ctime = time()
            print("%s/%s (%.02f%%) ~ %.02f (eta %.02f) h"
                  % (str(ii).zfill(zfill_index), str(nMC).zfill(zfill_index),
                     (ii / nMC * 100.0), ((ctime - t0) / 3600.0),
                     ((ctime - t0) / 3600.0 * nMC / (ii + 1))))
            sys.stdout.flush()

        if save_all_energies:
            all_e[ii, :] = e0

        # Generate a new configuration.
        xf = sample_nball(N, nMC=1, n_vec=n_vec)
        ef = energy(xf, lmbd=lambd_, ptype=ptype)
        deltaE = ef - e0

        # "Particles" in which dE < 0.
        dE_down = np.where(deltaE < 0.0)[1]

        # Compute the Boltzmann factor.
        rand_vec = np.random.random(size=(1, n_vec))
        w_vec = np.exp(-beta * deltaE)

        # Similarly, moves accepted to go up, or move rejected entirely.
        dE_up = np.where((deltaE >= 0.0) & (rand_vec <= w_vec))[1]
        dE_stay = np.where((deltaE >= 0.0) & (rand_vec > w_vec))[1]

        # Update the xf vector where necessary.
        xf[:, :, dE_stay] = x0[:, :, dE_stay]
        ef[:, dE_stay] = e0[:, dE_stay]

        # Append the basin / config. First, the configurations in which a move
        # was rejected stay in the same configuration, so those entries += 1
        # in the n_config counter.
        n_config[dE_stay] += 1

        # Get the order of magnitudes of the length of time remaining in a
        # given configuration.
        config_up = n_config[dE_up]
        config_down = n_config[dE_down]
        counter_up = Counter(int(floor(log2(x__))) for x__ in config_up)
        counter_down = Counter(int(floor(log2(x__))) for x__ in config_down)

        # Tricky dictionary manipulation with the Counter class, keys are the
        # order of magnitude, values are added between dictionaries (or
        # created if they did not exist).
        psi_config = psi_config + counter_up + counter_down

        # Then reset the counters after logging them.
        n_config[dE_up] = 1
        n_config[dE_down] = 1

        # Next, update the basin counter. First, determine configurations that
        # have entered a basin.
        entered_basin = np.where((ef < e_threshold) & (e0 >= e_threshold))[1]

        # Also, determine the configurations that were in a basin, and still
        # in the basin.
        still_in_basin = np.where((ef < e_threshold) & (e0 < e_threshold))[1]

        # In these two cases, add to counter.
        n_basin[entered_basin] += 1
        n_basin[still_in_basin] += 1

        # Now there are two cases. Note that if e0, ef > e_threshold, there's
        # nothing to do, as this configuration, and its prior one lie above the
        # cutoff. However, if a configuration exits a basin, meaning that
        # e0 < e_threshold and ef >= e_threshold,  it needs to be recorded.
        exited_basin = np.where((e0 < e_threshold) & (ef >= e_threshold))[1]

        exited_basin_cc = n_basin[exited_basin]
        basin_log = Counter(int(floor(log2(x__))) for x__ in exited_basin_cc)
        psi_basin += basin_log

        # And reset.
        n_basin[exited_basin] = 0

        # Now the xf's become the x0's for the next iteration.
        x0[:, :, dE_down] = xf[:, :, dE_down]
        x0[:, :, dE_up] = xf[:, :, dE_up]
        e0[:, dE_down] = ef[:, dE_down]
        e0[:, dE_up] = ef[:, dE_up]

        # Get the average energy if the timestep warrents it.
        if ii in sample_e:
            avg_e.append(np.mean(e0))

    total_time = ((time() - t0) / 3600.0)

    # Reset the overflow warnings.
    np.seterr(over='warn')

    pickle.dump(avg_e, open(os.path.join(data_directory, "avg_e.pkl"), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(sample_e,
                open(os.path.join(data_directory, "sample_e.pkl"), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(psi_config, open(os.path.join(data_directory,
                                              "psi_config.pkl"), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(psi_basin,
                open(os.path.join(data_directory, "psi_basin.pkl"), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    if save_all_energies:
        pickle.dump([all_e],
                    open(os.path.join(data_directory, "all_e.pkl"), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    print("Done. %.05f h" % total_time)
