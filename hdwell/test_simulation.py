#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone & Marco Baity-Jesi"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

"""Testing suite, for fun!"""

import numpy as np

from .auxiliary.aux import potential_surface, accept_canonical, delta_E


BETA = 100.0


def test_metropolis(nMC, N, dim=3):
    x0 = np.random.normal(scale=1.0, size=(N, dim))
    e0 = potential_surface(x0)

    positions = np.empty((N, dim, nMC))
    energy = []
    accepted = 0

    for ii in range(nMC):

        # Select a particle from the ensemble at random.
        random_particle = np.random.randint(0, N)

        # Generate some normal displacement delta.
        delta = np.random.normal(scale=0.1, size=(1, dim))

        # Push one particle by delta.
        xf = x0.copy()
        xf[random_particle] = xf[random_particle] + delta

        # Calculate final energies and acceptance probabilities.
        ef = potential_surface(xf)
        ww = accept_canonical(BETA, delta_E(e0, ef))

        # Acceptance criteria:
        if ww > 1.0 or np.random.random() < ww:
            positions[:, :, ii] = xf
            energy.append(ef)
            x0 = xf
            e0 = ef
            accepted += 1

        # Else, reject.
        else:
            positions[:, :, ii] = x0
            energy.append(e0)

    print("%.02f of moves accepted" % (100.0 * accepted / nMC))
    return [positions, energy]
