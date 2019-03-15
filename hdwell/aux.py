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
from scipy.integrate import quad

from . import logger  # noqa
lg = logging.getLogger(__name__)


AVAILABLE_P_TYPES = ['quad', 'log']
ENERGY_SAMPLE = 1000
MEM_SAMPLE = 50
DELTA_OMEGA = 0.5


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


def hxw(x, w):
    """Equation 3 in the Cammarota 2018 paper."""

    def integrand(u):
        return 1.0 / (1.0 + u) / u**x

    return quad(integrand, w, np.inf)[0] * np.sin(np.pi * x) / np.pi


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
                     data_directory, save_all_energies, save_all_stats,
                     verbose=True):
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
    save_all_stats : bool
        Whether or not to save all statistical information so that averages may
        be computed later (via concatenating many runs with the same
        parameters; this allows for further parallelization). This will save
        the following statistical information to disk:
            * Energy for every clone, at each point at which we sample it
            * Same for the average minimum radius

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
    min_r = np.ones((n_vec))
    avg_min_r = []

    # Initialize the number of timesteps in the current basin / config. Each
    # time a "particle" changes basins / configuration, this counter will
    # reset.
    n_basin = np.zeros((n_vec))
    n_config = np.ones((n_vec))

    # Some particles will already be in a basin.
    n_basin[np.where(e0 < e_threshold)[1]] += 1

    # Timesteps at which to sample the energy.
    sample_e = \
        np.unique(np.logspace(0, nMC_lg, ENERGY_SAMPLE, dtype=int,
                              endpoint=True))

    # Initialize the appropriate objects for saving everything.
    if save_all_stats:
        save_all_stats_energy = np.empty((len(sample_e), n_vec))
        save_all_stats_min_avg_r = np.empty((len(sample_e), n_vec))

    # We will now define the points at which to determine Pi_config and
    # Pi_basin. Starting with some defined DELTA_OMEGA, and the number of MC
    # points, this defines the maximum point at which we can sample on the
    # first grid via the second argument in Pi: tw + tw * DELTA_OMEGA.
    tw_max = floor(nMC / (DELTA_OMEGA + 1))

    # The first grid then is:
    pi_grid_sample_1 = np.unique(np.logspace(0, log10(tw_max), MEM_SAMPLE,
                                             dtype=int, endpoint=True))

    # The second grid maps exactly to the first grid via
    # tw += tw * DELTA_OMEGA.
    pi_grid_sample_2 = (pi_grid_sample_1 * (DELTA_OMEGA + 1.0)).astype(int)
    np.testing.assert_equal(True, pi_grid_sample_2[-1] <= nMC)

    # The number of points in these pi_grids:
    N_pi_grid = len(pi_grid_sample_2)
    np.testing.assert_equal(N_pi_grid, len(pi_grid_sample_1))

    # Now we need a matrix which saves the full configuration at each recorded
    # point so later they may be compared; one for the configs, one for the
    # basins. Note that if we wish to do this for more DELTA_OMEGA's we need
    # many more of these matrices. This may need to be rethought if we need
    # many more DELTA_OMEGA's!
    basin_recorder1 = np.zeros((N_pi_grid, n_vec), dtype=complex)
    basin_recorder2 = np.zeros((N_pi_grid, n_vec), dtype=complex)
    config_recorder1 = np.zeros((N_pi_grid, n_vec), dtype=complex)
    config_recorder2 = np.zeros((N_pi_grid, n_vec), dtype=complex)

    # To detemrine which basin the "particle" is in at any given time, we need
    # to keep a counter. It will work as follows: each particle in the ensemble
    # will be assigned a "basin index." The basin index will take on a real
    # value corresponding to the basin number it is in. In other words, in the
    # first basin, this particle will have basin index = 1. Once it exits the
    # first basin, it acquires imaginary component i. In this way, we can keep
    # track of whether or not the particle is in a basin, but which basin.
    # Furthermore, once that particle enters the next basin, it looses the
    # imaginary component, and gets +1 to its real component. Note that at
    # first we assume every particle is outside a basin.
    basin_index = np.ones((n_vec), dtype=complex) * 1j

    # Initialize the energy vector if report save all energies is true.
    if save_all_energies:
        all_e = np.empty((nMC, n_vec))

    # Temporarily ignore overflow warnings, since exp(large number) = inf and
    # this is fine for the code.
    np.seterr(over='ignore')

    # Begin the MC process.
    counter = 0
    counter1 = 0
    counter2 = 0
    counter_save_all_stats_E = 0
    counter_save_all_stats_R = 0
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

        # Update the minimum r vector
        rf = np.sqrt(np.sum(xf**2, axis=1)).squeeze()
        to_update = np.where(rf < min_r)[0]
        min_r[to_update] = rf[to_update]

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

        # Account for *which* basin the particles are in. Any particle that
        # just entered a basin gets 1-i:
        basin_index[entered_basin] += (1 - 1j)

        # Any particle which remains in a basin does not change. However, the
        # basins where a particle just left acquires an imaginary component:
        basin_index[exited_basin] += 1j

        # If we hit a point to record, update the configuration, add that
        # configuration to the recorder.
        if ii in pi_grid_sample_1:
            config_recorder1[counter1, :] = rf
            basin_recorder1[counter1, :] = basin_index
            counter1 += 1
        if ii in pi_grid_sample_2:
            config_recorder2[counter2, :] = rf
            basin_recorder2[counter2, :] = basin_index
            counter2 += 1

        # Now the xf's become the x0's for the next iteration.
        x0[:, :, dE_down] = xf[:, :, dE_down]
        x0[:, :, dE_up] = xf[:, :, dE_up]
        e0[:, dE_down] = ef[:, dE_down]
        e0[:, dE_up] = ef[:, dE_up]

        # Get the average energy if the timestep warrents it. Also update the
        # average minimum radius.
        if ii in sample_e:
            avg_e.append(np.mean(e0))
            avg_min_r.append(np.mean(rf))
            counter += 1
            if save_all_stats:
                save_all_stats_energy[counter_save_all_stats_E, :] = e0
                save_all_stats_min_avg_r[counter_save_all_stats_R, :] = min_r
                counter_save_all_stats_E += 1
                counter_save_all_stats_R += 1

    total_time = ((time() - t0) / 3600.0)

    # Reset the overflow warnings.
    np.seterr(over='warn')

    np.testing.assert_equal(counter, len(avg_e))
    np.testing.assert_equal(counter1, N_pi_grid)
    np.testing.assert_equal(counter2, N_pi_grid)

    basin_recorder = basin_recorder1 == basin_recorder2
    n_basin_output = np.sum(basin_recorder, axis=1)
    config_recorder = config_recorder1 == config_recorder2
    n_config_output = np.sum(config_recorder, axis=1)

    pickle.dump(avg_e, open(os.path.join(data_directory, "avg_e.pkl"), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(sample_e,
                open(os.path.join(data_directory, "sample_e.pkl"), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(psi_config,
                open(os.path.join(data_directory, "psi_config.pkl"), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(psi_basin,
                open(os.path.join(data_directory, "psi_basin.pkl"), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([n_basin_output, pi_grid_sample_1, DELTA_OMEGA],
                open(os.path.join(data_directory, "memory_basin.pkl"), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([n_config_output, pi_grid_sample_1, DELTA_OMEGA],
                open(os.path.join(data_directory, "memory_config.pkl"), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(avg_min_r,
                open(os.path.join(data_directory, "avg_min_r.pkl"), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

    if save_all_energies:
        pickle.dump(all_e,
                    open(os.path.join(data_directory, "all_e.pkl"), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    if save_all_stats:
        pickle.dump(save_all_stats_energy,
                    open(os.path.join(data_directory, "sas_all_e.pkl"), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(save_all_stats_min_avg_r,
                    open(os.path.join(data_directory,
                                      "sas_min_avg_r_e.pkl"), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(config_recorder,
                    open(os.path.join(data_directory,
                                      "sas_psi_config.pkl"), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(basin_recorder,
                    open(os.path.join(data_directory,
                                      "sas_psi_basin.pkl"), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    print("Done. %.05f h" % total_time)
