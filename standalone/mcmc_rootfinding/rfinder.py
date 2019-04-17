#!/usr/bin/env python3

__author__ = "Matthew Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"


import multiprocessing
import numpy as np
import yaml
from time import time, sleep
import pickle
import sys
import math


def nball_sampling(N, nMC=1):
    x_vector = np.random.normal(scale=1.0 / np.sqrt(2), size=(nMC, N))
    sum_of_squares = np.sum(x_vector**2, axis=1, keepdims=True)
    y = np.random.exponential(scale=1.0, size=(nMC, 1))
    vec_final = x_vector / np.sqrt(sum_of_squares + y)
    return vec_final


def monte_carlo_pDOWN_EXP(bigR, delta, N, nMC=1000, seed=None):

    if seed is not None:
        np.random.seed(seed)

    # This vector describes the distribution of the RADIUS
    vecR = np.random.exponential(scale=1.0 / delta, size=(nMC, 1))

    # This vector is the UNIFORM distribution of points on the hypersurface.
    # See Muller 1959, Marsaglia 1972,
    # http://mathworld.wolfram.com/HyperspherePointPicking.html
    vecX = np.random.normal(scale=1.0, size=(nMC, N))

    # Normalize it, then multiply by the length of the vector:
    vecX = vecX / np.sqrt(np.sum(vecX**2, axis=1, keepdims=True)) * vecR
    vecX[:, 0] += bigR  # shift the center

    return np.sum(np.sum(vecX**2, axis=1) < bigR**2) / nMC


def monte_carlo_pUP_EXP(bigR, delta, N, beta, nMC=1000, seed=None):

    if seed is not None:
        np.random.seed(seed)
    # Get the initial point
    vecX0 = nball_sampling(N, nMC=nMC)
    vecX0 = vecX0 / np.sqrt(np.sum(vecX0**2, axis=1, keepdims=True)) * bigR

    # This vector describes the distribution of the RADIUS
    vecX_R = np.random.exponential(scale=1.0 / delta, size=(nMC, 1))

    vecX = nball_sampling(N, nMC=nMC)

    # Normalize it, then multiply by the length of the vector:
    vecX = vecX / np.sqrt(np.sum(vecX**2, axis=1, keepdims=True))
    vecX = vecX * vecX_R
    vecX_final = vecX0 + vecX  # Displacement

    r0 = np.sqrt(np.sum(vecX0**2, axis=1))
    r = np.sqrt(np.sum(vecX_final**2, axis=1))
    xx = (r / r0)**(-beta * N)
    xx[xx > 1.0] = 1.0
    xp = np.random.random(size=r.shape) < xx
    s1 = bigR
    s2 = np.sqrt(np.sum(vecX_final**2, axis=1))

    return np.sum((s1 < s2) & (s2 < 1.0) & xp) / nMC


def generate_grids(p):

    if p['delta']['log_grid']:
        delta_grid = \
            np.logspace(p['delta']['minimum'], p['delta']['maximum'],
                        p['delta']['n'])
    else:
        delta_grid = \
            np.linspace(p['delta']['minimum'], p['delta']['maximum'],
                        p['delta']['n'])

    if p['beta']['log_grid']:
        beta_grid = np.unique(
            np.concatenate(
                (np.logspace(p['beta']['minimum'], 0,
                             int(p['beta']['n'] / 2)),
                 np.logspace(0, p['beta']['maximum'],
                             int(p['beta']['n'] / 2)))))
    else:
        beta_grid = np.unique(
            np.concatenate(
                (np.linspace(p['beta']['minimum'], 1.0,
                             int(p['beta']['n'] / 2)),
                 np.linspace(1.0, p['beta']['maximum'],
                             int(p['beta']['n'] / 2)))))

    if p['dim']['log_grid']:
        dim_grid = \
            np.unique(
                np.logspace(p['dim']['minimum'], p['dim']['maximum'],
                            p['dim']['n'], dtype=int))
    else:
        dim_grid = np.linspace(p['dim']['minimum'], p['dim']['maximum'],
                               p['dim']['n'])

    if p['x0']['log_grid']:
        x0_grid = np.logspace(p['x0']['minimum'], p['x0']['maximum'],
                              p['x0']['n'], endpoint=False)
    else:
        x0_grid = np.linspace(p['x0']['minimum'], p['x0']['maximum'],
                              p['x0']['n'], endpoint=False)

    return delta_grid, beta_grid, dim_grid, x0_grid


if __name__ == '__main__':
    params = yaml.safe_load(open("params.yaml"))
    nmc = int(params['nmc'])

    print("Parameters --------------------------------")
    for key, value in params.items():
        print(key, "=", value)

    n_workers = multiprocessing.cpu_count()
    print(n_workers, 'cpus')

    delta_grid, beta_grid, dim_grid, x0_grid = generate_grids(params)
    n_beta, n_delta, n_x0 = len(beta_grid), len(delta_grid), len(x0_grid)
    n_dim = len(dim_grid)
    print("New delta_grid len:   %i" % n_delta)
    print("New beta_grid len:    %i" % n_beta)
    print("New dim_grid len:     %i" % n_dim)
    print("New x0g_grid len:     %i" % n_x0)

    zf1 = int(math.floor(math.log(n_dim, 10)))
    zf2 = int(math.floor(math.log(n_beta, 10)))
    zf3 = int(math.floor(math.log(n_delta, 10)))
    zf4 = int(math.floor(math.log(n_x0, 10)))

    def submit(counter, ii, dim, jj, beta, kk, delta, ll, x0, r_dict):
        dim = int(dim)
        up = monte_carlo_pUP_EXP(x0, delta, dim, beta, nMC=nmc)
        down = monte_carlo_pDOWN_EXP(x0, delta, dim, nMC=nmc)
        r_dict[counter] = [ii, jj, kk, ll, up, down]
        print("%s ~ %s ~ %s ~ %s : %.05f ~ %.05f DONE"
              % (str(ii).zfill(zf1), str(jj).zfill(zf2), str(kk).zfill(zf3),
                 str(ll).zfill(zf4), down, up))
        sys.stdout.flush()

    processes = []
    manager = multiprocessing.Manager()
    r_dict = manager.dict()

    t0 = time()
    counter = 0

    pool = multiprocessing.Pool(n_workers)
    for ii, dim in enumerate(dim_grid):
        for jj, beta in enumerate(beta_grid):
            for kk, delta in enumerate(delta_grid):
                for ll, x0 in enumerate(x0_grid):
                    print("%s ~ %s ~ %s ~ %s : SUB"
                          % (str(ii).zfill(zf1), str(jj).zfill(zf2),
                             str(kk).zfill(zf3), str(ll).zfill(zf4)))
                    # p = multiprocessing.Process(
                    #     target=submit,
                    #     args=(counter, ii, dim, jj, beta, kk, delta, ll, x0,
                    #           r_dict))

                    pool.apply_async(submit,
                                     args=(counter, ii, dim, jj, beta, kk,
                                           delta, ll, x0, r_dict))
                    counter += 1
                    # processes.append(p)
                    # p.start()
                    sys.stdout.flush()

    pool.close()
    pool.join()
    sys.stdout.flush()

    d = r_dict.values()

    MAT_UP = np.empty((n_dim, n_beta, n_delta, n_x0))
    MAT_DOWN = np.empty((n_dim, n_beta, n_delta, n_x0))
    for ii, value in enumerate(d):
        MAT_UP[value[0], value[1], value[2], value[3]] = value[4]
        MAT_DOWN[value[0], value[1], value[2], value[3]] = value[5]

    pickle.dump(MAT_UP, open("pUP.pkl", 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(MAT_DOWN, open("pDOWN.pkl", 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

    print("total time elapsed:", np.round((time() - t0) / 60., 3), "m")
