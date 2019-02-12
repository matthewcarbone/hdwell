#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone & Marco Baity-Jesi"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

"""Testing suite, for fun!"""

import numpy as np
from tqdm import tqdm

from .auxiliary.aux import potential_surface, accept_canonical, delta_E


BETA = 100.0


def test_metropolis(nMC, N=1, dim=3, default_scale=1.0, delta_scale=0.5):
    x0 = np.random.normal(scale=default_scale, size=(N, dim))
    e0 = potential_surface(x0, dim)

    positions = np.empty((N, dim, nMC))
    energy = []
    accepted = 0

    for ii in tqdm(range(nMC)):

        # Select a particle from the ensemble at random.
        random_particle = np.random.randint(0, N)

        if N == 1:
            np.testing.assert_equal(random_particle, 0)

        # Generate some normal displacement delta.
        delta = np.random.normal(scale=delta_scale, size=(1, dim))

        # Push one particle by delta.
        xf = x0.copy()
        xf[random_particle] = xf[random_particle] + delta

        # Calculate final energies and acceptance probabilities.
        ef = potential_surface(xf, dim)
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

    print("%.02f of moves accepted for N=%i, dim=%i"
          % ((100.0 * accepted / nMC), N, dim))
    return [positions, energy]
