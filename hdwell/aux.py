#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone & Marco Baity-Jesi"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

"""All auxiliary/support functions."""

import numpy as np
import os
from math import floor, log10
import logging
import datetime
import pickle
from time import time
from itertools import product
from collections import Counter

from . import logger  # noqa
lg = logging.getLogger(__name__)


AVAILABLE_P_TYPES = ['quad', 'log']


def order_of_magnitude(x):
    """Returns the base 10 order of magnitude of `x`."""

    return int(log10(x))


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
    dt = now.strftime("%Y-%m-%d-%H-%M")
    lg.info("Datetime logged as %s" % dt)
    return dt


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

    and returns all permutations:

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
    """Returns the threshold radius and energy."""

    if beta == N:
        r = np.exp(-1.0 / N)
    else:
        r = (2.0 - beta / N)**(1.0 / (beta - N))

    if ptype == 'log':
        e = N * np.log(r) / lambdaprime
    else:
        raise RuntimeError("ptype %s not supported." % ptype)

    return [r, e]


def pure_mc_sampling(N, beta, lambdaprime, nMC, n_vec, ptype, obs, n_report,
                     data_directory):
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
    nMC : int
        Total number of monte carlo timesteps to perform.
    n_vec : int
        Total number of "particles" in the "ensemble".
    ptype : str
        Determines the energy of the system (fixes the form of the potential
        energy).
    obs : dict
        Python dictionary of boolean values and keys corresponding to which
        observables to report. Generally, we should report all observables,
        as there is no extra overhead.
    n_report : int
        Report progress (pipe to output) everytime the simulation reaches this
        many timesteps.
    data_directory : str
        Absolute path to the location of the data directory to which
        observables will be saved.
    """

    nMC = 10**nMC
    lambd_ = lambdaprime / N
    increment = int(nMC / n_report)  # TODO: if increment is 0
    zfill_index = order_of_magnitude(nMC) + 1
    [r_threshold, e_threshold] = thresholds(N, beta, lambdaprime, ptype=ptype)

    x0 = sample_nball(N, nMC=1, n_vec=n_vec)  # Get the initial position
    e0 = energy(x0, lmbd=lambd_, ptype=ptype)    # Initial energy
    t0 = time()                                 # Initial time

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
    sample_e = np.logspace(0, nMC, nMC + 1, dtype=int)

    # Begin the MC process.
    for ii in range(1, 100):

        # Report at designated timesteps.
        if ii % increment == 0:
            ctime = time()
            print("%s/%s (%.02f%%) ~ %.02f (eta %.02f) h"
                  % (str(ii).zfill(zfill_index), str(nMC).zfill(zfill_index),
                     (ii / nMC * 100.0), ((ctime - t0) / 3600.0),
                     ((ctime - t0) / 3600.0 * nMC / ii)))

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

        # Append the basin / config. First, the configurations in which a move
        # was rejected stay in the same configuration, so those entries += 1
        # in the n_config counter.
        n_config[dE_stay] += 1

        # Get the order of magnitudes of the length of time remaining in a
        # given configuration.
        config_up = n_config[dE_up]
        config_down = n_config[dE_down]
        lg.info("%a" % config_up)
        lg.info("%a" % config_down)
        counter_up = Counter(int(floor(log10(x__))) for x__ in config_up)
        counter_down = Counter(int(floor(log10(x__))) for x__ in config_down)

        # Tricky dictionary manipulation with the Counter class, keys are the
        # order of magnitude, values are added between dictionaries (or
        # created if they did not exist).
        psi_config += counter_up + counter_down

        # Then reset the counters after logging them.
        n_config[dE_up] = 1
        n_config[dE_down] = 1

        # Next, update the basin counter. First, determine which new energies
        # are below the threshold.
        below = np.where(ef < e_threshold)[1]

        # Wherever the energy is below threshold, add to counter.
        n_basin[below] += 1

        # Now there are two cases. If the prior energy was in a basin but just
        # exited, we need to log it.
        exited_basin = np.where((e0 < e_threshold) & (ef >= e_threshold) &
                                (n_basin != 0))[1]
        exited_basin_cc = n_basin[exited_basin]
        basin_log = Counter(int(floor(log10(x__))) for x__ in exited_basin_cc)
        psi_basin += basin_log

        # And reset.
        n_basin[exited_basin] = 0

        # Update.
        x0[:, :, dE_down] = xf[:, :, dE_down]
        x0[:, :, dE_up] = xf[:, :, dE_up]
        e0[:, dE_down] = ef[:, dE_down]
        e0[:, dE_up] = ef[:, dE_up]

        # Get the average energy if the timestep warrents it.
        if ii in sample_e:
            avg_e.append(np.mean(ef))

    pickle.dump(avg_e, open(os.path.join(data_directory, "avg_e.pkl"), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(psi_config, open(os.path.join(data_directory,
                                              "psi_config.pkl"), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(psi_basin,
                open(os.path.join(data_directory, "psi_basin.pkl"), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
