#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

"""Core simulation."""

import numpy as np
import sys
from time import time
from collections import Counter
from math import floor
import pickle

import hdwell2.auxiliary as aux

HP = pickle.HIGHEST_PROTOCOL


def simulate(nMC_lg, N, n_tracer, beta, delta, ptype, protocol,
             save_all_energies=False, n_report=1000, verbose=True,
             lambdaprime=1.0, dw=0.5, n_e_sample=1000, n_mem_sample=50):
    """Executes a MC simulation over the unit N-ball. Uses Metropolis selection
    criteria.

    Parameters
    ----------
    nMC_lg : int
        Total number of monte carlo timesteps to perform is 10^nMC_lg.
    N : int
        Dimension of the system where the tracers evolve.
    n_tracer : int
        Total number of tracers in the ensemble.
    beta : float
        Inverse temperature 1/T.
    delta : float
        Scale parameter which controls the scaling of the radius during Markov-
        chain MC.
    ptype : str
        Determines the energy of the system (fixes the form of the potential
        energy).
    n_report : int
        Report progress (pipe to output) everytime the simulation reaches this
        many timesteps.
    protocol : int
        Defines the simulation type.
            * 0 => DSMC
            * 1 => MCMC
    save_all_energies : bool
        If true, will save the 10^nMC-dimensional vector containing the average
        energies at every timestep. Mainly used for debugging, since for any
        large number of timesteps this is guaranteed to kill your machine. For
        this reason, defaults to False.
    verbose : bool
        Whether or not to pipe current status at increments to console. Default
        is True.
    lambdaprime : float
        Scale parameter set as lambdaprime = lambda * N. Generally, the
        potential will be defined something like lambda^{-1}*ln r, and this
        lambdaprime is fixed such that the DOS of our system matches that of
        literature precedent. Ultimately, lambdaprime=1 is a good default and
        probably should not be changed. This scaling also guarantees an
        extensive energy with respect to the dimension.
    dw : float
        Traping dynamics parameter (see Cammarota 2018).
    n_e_sample : int
        Number of points where we sample the energy.
    n_mem_sample : int
        Number of points where we sample the memory functions Pi.

    Returns
    -------
    tau_sample_grid : list
        List of points where the timestep was sampled (logarithmic)
    eAVG, eSTD, rminAVG, rminSTD: list
        List of points representing the average (and standard deviation) of the
        energy and minimum radius.
    psiB, psiC : collections.Counter
        Counters for the psi basin and config where the key indexes the
        log2(n_points), where n_points is the number of timesteps that the
        tracer remained in some basin/configuration.
    pi_grid_sample_1 : array like
        Numpy array (1d) of the grid points where pi was sampled (tw, not
        tw + tw * dw).
    Pi_basin_final_output, Pi_config_final_output : array like
        Probabilities, properly normalized, corresponding to Pi
        basin/configuration.
    """

    if protocol == 1 and delta is None:
        raise RuntimeError("Need to initialize delta with MCMC simulation.")

    nMC = int(10**nMC_lg)  # Define the total number of MC time steps

    if verbose:
        print("NMC: %i" % nMC)

    # With lambda' = lambda * N, we constrain lambda and choose N such that
    # lambda' is fixed to be the critical beta which further constrains
    # (for log potential) E = ln r / lambda. Usually we choose lambda_prime
    # to be 1.
    lambda_ = lambdaprime / N

    if verbose:
        print("LAMBDA: %.02f" % lambda_)

    # Used when saving files (determines the number of padding 0's when saving
    # a text file to disk, for example).
    zfill_index = aux.order_of_magnitude(nMC) + 1

    # Get the threshold information, which is only a function of the
    # dimensionality, N, and the inverse temperature (as well as the scale
    # factor, lambdaprime, but this is unimportant). Note that lambdaprime
    # is infact equal to the critical inverse temperature the way we chose it.
    [r_th, e_th] = aux.thresholds(N, beta, lambdaprime, ptype=ptype)

    if verbose:
        print("r/e_th: %.02f/%.02f" % (r_th, e_th))

    # Initialize a uniform distribution on the N-ball
    x0 = aux.sample_nball(N, n_tracer)

    # Get the corresponding initial energes
    e0 = aux.energy(x0, lambda_, ptype)

    # Wall clock time
    t0 = time()

    # Initialize an average energy vector which will keep track of the energy
    # of the tracers, averaged over all tracers, for every MC timestep,
    # henceforth referred to as tau. Also keep track of the standard deviation
    # of the data.
    eAVG = []
    eSTD = []

    # The persistence functions (psi) are binned logarithmically; a Counter
    # class is optimal to keep track of these, with the keys of the Counter
    # referencing the number of times that timeframe is observed.
    psiB = Counter({})  # Basin
    psiC = Counter({})  # Config

    # It is helpful to keep track of the minimum obtained radius for each
    # tracer,
    min_r_for_tracer = np.ones((n_tracer))

    # as well as the average over all tracers as a function of the timestep,
    # sampled at the same intervals as the energy. Similar to the energy, we
    # also want error bars.
    rminAVG = []
    rminSTD = []

    # For every tracer, these counters keep track of the amount of time steps
    # that a tracer remains in a single basin or configuration. Each time
    # a tracer leaves the basin or configuration, the counter resets.
    n_tau_in_B = np.zeros((n_tracer))
    n_tau_in_C = np.ones((n_tracer))

    # Initialize the timesteps where the energy / min radius (and other
    # quantities) are sampled at.
    tau_sample_grid = []

    # The memory functions (Pi) need to be sampled at tau-points tw and
    # tw + tw * dw, where dw is a parameter corresponding to the hwx function
    # in auxiliary.py. We will now define the points at which to determine Pi_C
    # and Pi_B. Starting with some defined dw, and the number of MC
    # points, this defines the maximum point at which we can sample on the
    # first grid such that the second argument in Pi: tw + tw * dw exists in
    # the MC sampling procedure.
    tw_max = nMC // (dw + 1.0)

    # Using this value allows us to define the first grid. Note it is not
    # always the case, depending on the amount of points in the grid, that
    # the length of this grid will = n_mem_sample, although it will be close.
    pi_grid_sample_1 = np.unique(np.logspace(0, np.log10(tw_max), n_mem_sample,
                                             dtype=int, endpoint=True))

    # The second grid is related directly to the first via tw -> tw + tw * dw,
    pi_grid_sample_2 = (pi_grid_sample_1 * (dw + 1.0)).astype(int)
    np.testing.assert_equal(True, pi_grid_sample_2[-1] <= nMC)
    N_pi_grid = len(pi_grid_sample_1)
    np.testing.assert_equal(N_pi_grid, len(pi_grid_sample_2))

    # Now we need a matrix which saves the full configuration at each recorded
    # point so later they may be compared; one for the configs, one for the
    # basins.
    B_recorder1 = np.zeros((N_pi_grid, n_tracer), dtype=complex)
    B_recorder2 = np.zeros((N_pi_grid, n_tracer), dtype=complex)
    C_recorder1 = np.zeros((N_pi_grid, n_tracer))
    C_recorder2 = np.zeros((N_pi_grid, n_tracer))

    # To determine which basin the "particle" is in at any given time, we need
    # to keep a counter. It will work as follows: each particle in the ensemble
    # will be assigned a "basin index." The basin index will take on a real
    # value corresponding to the basin number it is in. In other words, in the
    # first basin, this particle will have basin index = 1. Once it exits the
    # first basin, it acquires imaginary component i. In this way, we can keep
    # track of not just whether or not the particle is in a basin, but which
    # basin. Furthermore, once that particle enters the next basin, it looses
    # the imaginary component, and gets +1 to its real component.
    basin_index = np.ones((n_tracer), dtype=complex) * 1j

    # Optional: even at the start, particles that are < e_threshold are
    # considered to be in a basin.
    in_basin_to_start = np.where(e0 < e_th)[1]
    basin_index[in_basin_to_start] = 0
    n_tau_in_B[in_basin_to_start] = 1

    # It is also critical to have a normalization term for the following
    # reason. A tracer at tw that is not in a basin should not contribute to
    # the overall probability embodied by Pi_B, which states: "given that a
    # tracer is in a basin, the tracer is still in that basin at tw + tw * dw."
    # This normalizer will count the number of tracers which at tw, are in a
    # basin/config.
    B_normalization = np.zeros((N_pi_grid))

    # Optional for debugging: save all energies for EVERY tracer, at every
    # timestep:
    if save_all_energies:
        eALL = np.empty((nMC, n_tracer))

    # Get the increments where we should sample the energy (this is
    # logarithmic)
    where_sample_e = \
        np.unique(np.logspace(0, nMC_lg, n_e_sample, dtype=int, endpoint=True))

    # And similarly for reporting (this is linear)
    report_increment = nMC // n_report

    # Temporarily ignore overflow warnings, since exp(large number) = inf and
    # this is fine for the code.
    np.seterr(over='ignore')

    # Initialize some counters
    counter1 = 0
    counter2 = 0

    # Begin the sampling.
    for ii in range(nMC + 1):

        # Report at designated timesteps so we can keep track of the
        # calculation.
        if ii % report_increment == 0 and verbose:
            ctime = time()
            print("%s/%s (%.02f%%) ~ %.02f (eta %.02f) h"
                  % (str(ii).zfill(zfill_index), str(nMC).zfill(zfill_index),
                     (ii / nMC * 100.0), ((ctime - t0) / 3600.0),
                     ((ctime - t0) / 3600.0 * nMC / (ii + 1))))
            sys.stdout.flush()

        if save_all_energies:
            eALL[ii, :] = e0

        # Generate a new configuration
        if protocol == 1:
            #                     --- Old Method ---
            # The random walk first samples a point randomly & uniformally on
            # the surface of the unit n-ball.
            # on_surface = sample_nball(N, nMC=1, n_vec=n_vec, yzero=True)

            # Next, a radius is sampled from an exponential distribution and
            # used to scale that ball.
            # exp_rand = np.random.exponential(scale=1.0 / xp_param,
            #                                  size=(1, 1, n_vec))

            # The `xf` point is then updated by walking the particle from the
            # old configuration `x0` to the new one.

            #                     --- New Method ---
            # Push the particle by a Gaussian-sampled number
            xf = x0 + np.random.normal(scale=delta, size=(x0.shape))
        elif protocol == 0:
            xf = aux.sample_nball(N, n_tracer)
        else:
            raise RuntimeError("Invalid protocol")

        # Get the energy of this new configuration
        ef = aux.energy(xf, lambda_, ptype=ptype)

        # Get the difference in energy between the old and new configs
        dE = ef - e0  # Of shape (N, n_tracer)

        # Compute the current distance of each particle from the center of the
        # well.
        rf = np.sqrt(np.sum(xf**2, axis=1)).squeeze()

        # Tracers which decrease in energy are accepted with p = 1:
        dE_down = np.where(dE < 0.0)[1]

        # Boltzmann factor:
        rand_vec = np.random.random(size=(1, n_tracer))
        w_vec = np.exp(-beta * dE)

        # Similarly, moves accepted to go up, or move rejected entirely.
        # For instance, a tracer can move up if dE > 0 but only if the
        # Metropolis selection criteria is satisfied.
        dE_up = np.where((dE >= 0.0) & (rand_vec <= w_vec))[1]

        # Any move in which dE >= 0 and the Boltzmann critiera is not
        # satisfied, or where the tracer's next move puts it outside the unit
        # ball, is rejected.
        dE_stay = np.where(((dE >= 0.0) & (rand_vec > w_vec)) | (rf > 1))[1]

        # Update the xf vector where necessary. All new moves (up or down) are
        # kept while re-initializing the required positions where no move has
        # take place, from the x0 vector.
        xf[:, :, dE_stay] = x0[:, :, dE_stay]
        ef[:, dE_stay] = e0[:, dE_stay]

        # Recompute the current radius after moves have been accepted/rejected
        rf = np.sqrt(np.sum(xf**2, axis=1)).squeeze()

        # Update the minimum r vector:
        to_update = np.where(rf < min_r_for_tracer)[0]
        min_r_for_tracer[to_update] = rf[to_update]

        # Easy to check n_config: only tracers that remained in the same
        # configuration (i.e. dE_stay) contribute:
        n_tau_in_C[dE_stay] += 1

        # Get the counters for the particles which exited up until this point:
        configUP = n_tau_in_C[dE_up]
        configDOWN = n_tau_in_C[dE_down]

        # And append this to the counter, which will bin logarithmically
        counter_up = Counter(int(floor(np.log2(x__))) for x__ in configUP)
        counter_down = Counter(int(floor(np.log2(x__))) for x__ in configDOWN)

        # Further append the frequency of these events to the overall counter
        psiC = psiC + counter_up + counter_down

        # Reset the counters
        n_tau_in_C[dE_up] = 1
        n_tau_in_C[dE_down] = 1

        # Next there's the basin counter. First, which configs enter a basin:
        entered_basin = np.where((ef < e_th) & (e0 >= e_th))[1]

        # And those that were in a basin, and are still in a basin:
        still_in_basin = np.where((ef < e_th) & (e0 < e_th))[1]

        # Those two cases acquire an increment to the real part, which indexes
        # the length of time that tracer has been in a basin.
        n_tau_in_B[entered_basin] += 1
        n_tau_in_B[still_in_basin] += 1

        # Now we count the tracers that have been in a basin previously, and
        # just exited.
        exited_basin = np.where((e0 < e_th) & (ef >= e_th))[1]

        # From there, we query the counter to determine how long that tracer
        # was in that basin, and index its log2 result:
        exited_basin_cc = n_tau_in_B[exited_basin]
        basin_log = \
            Counter(int(floor(np.log2(x__))) for x__ in exited_basin_cc)
        psiB = psiB + basin_log

        # And then reset the counter:
        n_tau_in_B[exited_basin] = 0

        # Account for *which* basin the particles are in. Any particle that
        # just entered a basin gets 1-i:
        basin_index[entered_basin] += (1 - 1j)

        # Any particle which remains in a basin does not change. However, the
        # basins where a particle just left acquires an imaginary component:
        basin_index[exited_basin] += 1j

        # If we hit a point in the Pi grid to record, do so:
        if ii in pi_grid_sample_1:

            # Note for C_recorder we simply save the radius, not the exact
            # configuration so that we can save a bit of space. The probability
            # of achieving a different configuraiton xf but the exact same
            # radius is ~0, so this should not be a problem.
            C_recorder1[counter1, :] = rf
            B_recorder1[counter1, :] = basin_index

            # Tricky part. We need to compute the normalization here.
            # Basically, if a a tracer is *not* in a basin, it will never
            # contribute to Pi basin calculated downstream, but it should also
            # *NOT* contribute to the normalization (the probability) overall.
            # Thus, here we compute the number of tracers which are actually
            # in a basin and store that value as the normaliztion.
            B_normalization[counter1] = np.sum(np.imag(basin_index) == 0)
            counter1 += 1

        if ii in pi_grid_sample_2:
            C_recorder2[counter2, :] = rf
            B_recorder2[counter2, :] = basin_index
            counter2 += 1

        # Now the xf's become the x0's for the next iteration.
        x0[:, :, dE_down] = xf[:, :, dE_down]
        x0[:, :, dE_up] = xf[:, :, dE_up]
        e0[:, dE_down] = ef[:, dE_down]
        e0[:, dE_up] = ef[:, dE_up]

        # Get the average energy if we're at a sampling timestep
        if ii in where_sample_e:

            # Energy
            eAVG.append(np.mean(e0))
            eSTD.append(np.std(e0))

            # Minimum radius
            rminAVG.append(np.mean(min_r_for_tracer))
            rminSTD.append(np.std(min_r_for_tracer))

            # Log the values of tau where we sampled these quantities
            tau_sample_grid.append(ii)

    total_time = ((time() - t0) / 3600.0)

    # Reset the overflow warnings
    np.seterr(over='warn')

    # Evaluate Pi basin: the first part ensures that the tracer has the same
    # basin index (real part) at both tw and tw + tw * dw. The second part
    # ensures that neither of the tracers happen to be out of a basin at the
    # time of recording. Finally, the third part accounts for the
    # normalization, meaning, any tracer which is not in a basin at tw does not
    # contribute to the overall probability ala Bayes rule.
    part1 = B_recorder1 == B_recorder2
    part2 = np.imag(B_recorder1) == 0
    _condition = (part1 & part2)
    basin_rec_final = np.sum(_condition, axis=1)
    Pi_basin_final_output = basin_rec_final / B_normalization

    # Do the same thing, essentially, for the config recorder:
    config_recorder = C_recorder1 == C_recorder2
    Pi_config_final_output = np.sum(config_recorder, axis=1)

    print("Done. %.05f h" % total_time)

    return [tau_sample_grid, eAVG, eSTD, rminAVG, rminSTD, psiB, psiC,
            pi_grid_sample_1, Pi_basin_final_output, Pi_config_final_output]


if __name__ == '__main__':

    # Execute the simulation (this should be run directly on a cluster)
    N_mc_LOG = int(sys.argv[1])
    N_dims = int(sys.argv[2])
    N_tracer = int(sys.argv[3])
    beta = float(sys.argv[4])
    delta = float(sys.argv[5])
    ptype = str(sys.argv[6])
    protocol = int(sys.argv[7])
    lambdaprime = float(sys.argv[8])
    dw = float(sys.argv[9])
    save_to = str(sys.argv[10])

    print(sys.argv)

    all_results = simulate(
        N_mc_LOG, N_dims, N_tracer, beta, delta, ptype, protocol,
        save_all_energies=False,
        n_report=1000,
        verbose=True,
        lambdaprime=lambdaprime,
        dw=dw,
        n_e_sample=1000,
        n_mem_sample=100)

    # Next save the results to disk:
    pickle.dump(all_results, open(save_to, 'wb'), protocol=HP)
