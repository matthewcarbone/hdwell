#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

"""Helper functions. Does not import any local modules."""

import numpy as np
from scipy.integrate import quad
from itertools import product
import os
import datetime


# General directory utilities

def makedir_if_not_exist(path, error_out=False):
    """Does exactly what is says it does: if the directory specified by `path`
    does not exist, create it."""

    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        if error_out:
            raise FileExistsError(  # noqa
                "%s exists already, will not overwrite." % path)


def current_datetime():
    """Get's the current date and time and converts it into a string to use
    for tagging files."""

    now = datetime.datetime.now()
    dt = now.strftime("%Y-%m-%d-%H-%M-%S")
    return dt


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
    return combinations


# Mathematical functions

def hxw(x, w):
    """Equation 3 in the Cammarota 2018 paper."""

    def integrand(u):
        return 1.0 / (1.0 + u) / u**x

    return quad(integrand, w, np.inf)[0] * np.sin(np.pi * x) / np.pi


def order_of_magnitude(x):
    """Returns the base 10 order of magnitude of `x`."""

    return int(np.log10(x))


def sample_nball(N, n_vec):
    """Uses the `smart` sampling method as described by Barthe, Guedon,
    Mendelson and Naor in Annals of Probability 33(2), 480 (2005). Samples on
    the unit N-ball uniformally. Note that the dimension of the returned
    NumPy array will be 1 x N x n_vec, since 1 indexes the MC timestep.

    Parameters
    ----------
    N : int
        Dimension of the system.
    n_vec : int
        Number of tracers in the ensemble.

    Returns a NumPy array which contains positions for every dimension,
    monte carlo timestep and particle.
    """

    x_vector = np.random.normal(scale=1.0 / np.sqrt(2.0), size=(1, N, n_vec))
    sum_of_squares = np.sum(x_vector**2, axis=1, keepdims=True)
    y = np.random.exponential(scale=1.0, size=(1, 1, n_vec))
    vec_final = x_vector / np.sqrt(sum_of_squares + y)
    return vec_final


def thresholds(N, beta, beta_c, ptype='log'):
    """Returns the threshold radius and energy.

    Parameters
    ----------
    N : int
        Dimension of the system.
    beta : float
        Inverse temperature 1/T.
    beta_c : float
        Critial inverse temperature.
    ptype : string
        Type of potential. Here we only have logarithmic base 2 available.

    Returns [r, e], where r (e) is the threshold radius (energy).
    """

    if ptype == 'log':
        if beta == 1.0:
            r = np.exp(-1.0 / N)
        elif beta >= 2.0 * beta_c:
            r = 0.0
        else:
            r = (2.0 - beta / beta_c)**(1.0 / (N * beta / beta_c - N))

        if beta >= beta_c:
            e = -np.inf
        else:
            e = N * np.log(r) / beta_c
    else:
        raise RuntimeError("ptype %s not supported." % ptype)

    return [r, e]


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

    if ptype == 'quad':
        return lmbd * np.sum(x**2)
    elif ptype == 'log':
        return np.log(np.sqrt(np.sum(x**2, axis=1))) / lmbd
    else:
        raise RuntimeError("Invalid potential")
