#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone & Marco Baity-Jesi"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

"""Contains all the potentials we may wish to try as well as any relevant
helper functions."""

import numpy as np


AVAILABLE_P_TYPES = ['quad']


def potential_surface(x, ptype='quad', gamma=1.0):
    """The potential energy as a function of some input `x`.

    Parameters
    ----------
    x : array_like
        Usually the positions of an ensemble of particles.
    ptype : str
        Available potential energies. Default is 'quad'. Available options:
        - 'quad' : Simple harmonic oscillator form v = gamma*x*x.
    gamma : float
        Multiplier parameter. Default is 1.0.
    """

    if ptype not in AVAILABLE_P_TYPES:
        raise RuntimeError("Fatal: invalid potential surface type `ptype`.")

    if ptype == 'quad':
        return gamma * np.sum(x**2)


def delta_E(e0, ef):
    """Returns the difference e_final - e_initial."""

    return ef - e0


def accept_canonical(beta, dE):
    """Acceptance criteria w = e^(-beta*(ef-e0)) as per the canonical
    ensemble."""

    return np.exp(-beta * dE)
