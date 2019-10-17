#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

"""Core simulation."""

import numpy as np
import pickle
import os
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


def hxw(x, w):
    """Equation 3 in the Cammarota 2018 paper."""

    def integrand(u):
        return 1.0 / (1.0 + u) / u**x

    return quad(integrand, w, np.inf)[0] * np.sin(np.pi * x) / np.pi


def plot_Pi(list_of_results, list_of_colors, list_of_betas, n_tracers,
            title=None, xscale='log', yscale=None, DFS=12, capsize=2,
            capthick=0.3, elw=0.3, marker='s', ms=1.0, ninc=30, lw=1.0,
            full_legend=False, dw=0.5):

    if title is not None:
        plt.title(r'%s' % title, fontsize=DFS)

    if xscale is not None:
        plt.xscale(xscale)

    if yscale is not None:
        plt.yscale(yscale)

    for jj in range(len(list_of_results)):

        if full_legend:
            label = r'$\Pi_{\mathrm{B}}(\tau),\: \beta=%.01f$' % \
                list_of_betas[jj]
        else:
            label = r'$\beta=%.01f$' % list_of_betas[jj]

        plt.errorbar(list_of_results[jj]['pi_grid'],
                     list_of_results[jj]['PiB'],
                     np.array(list_of_results[jj]['dPiB']) /
                     np.sqrt(n_tracers - 1), linewidth=lw,
                     color=list_of_colors[jj], marker=marker, ms=ms,
                     capthick=capthick, capsize=capsize, elinewidth=elw,
                     label=label)

        h = hxw(2.0 - list_of_betas[jj], dw)
        plt.plot(list_of_results[jj]['pi_grid'],
                 [h for __ in
                  range(len(list_of_results[jj]['pi_grid']))],
                 color=list_of_colors[jj], linestyle='--')


def plot_psi(list_of_results, list_of_colors, list_of_betas, n_tracers,
             title=None, xscale='log', yscale='log', DFS=12, capsize=2,
             capthick=0.3, elw=0.3, marker='s', ms=1.0, ninc=30, lw=1.0,
             full_legend=False):

    if title is not None:
        plt.title(r'%s' % title, fontsize=DFS)

    if xscale is not None:
        plt.xscale(xscale)

    if yscale is not None:
        plt.yscale(yscale)

    for jj in range(len(list_of_results)):

        ll = [2**ii for ii in range(len(list_of_results[jj]['psiB']))]

        if full_legend:
            label = r'$\psi_{\mathrm{B}}(\tau),\: \beta=%.01f$' % \
                list_of_betas[jj]
        else:
            label = r'$\beta=%.01f$' % list_of_betas[jj]

        plt.errorbar(ll, list_of_results[jj]['psiB'],
                     np.array(list_of_results[jj]['dpsiB']) /
                     np.array(list_of_results[jj]['psiB']) /
                     np.sqrt(n_tracers - 1) * 0.434, linewidth=lw,
                     color=list_of_colors[jj], marker=marker, ms=ms,
                     capthick=capthick, capsize=capsize, elinewidth=elw,
                     label=label)

        ll = [2**ii for ii in range(len(list_of_results[jj]['psiC']))]

        if full_legend:
            label = r'$\psi_{\mathrm{C}}(\tau),\: \beta=%.01f$' % \
                list_of_betas[jj]
        else:
            label = None

        plt.errorbar(ll, list_of_results[jj]['psiC'],
                     np.array(list_of_results[jj]['dpsiC']) /
                     np.array(list_of_results[jj]['psiC']) /
                     np.sqrt(n_tracers - 1) * 0.434, linewidth=lw,
                     linestyle='--', color=list_of_colors[jj], marker=marker,
                     ms=ms, capthick=capthick, capsize=capsize, elinewidth=elw,
                     label=label)


def plot_horizontal_line(x0, xf, y, cstyle, lw):
    """Plots a horizontal line."""

    plt.plot((x0, xf), (y, y), cstyle, linewidth=lw)


def plot_energy(list_of_results, list_of_colors, list_of_betas, n_tracers,
                title=None, xscale='log', yscale=None, DFS=12, capsize=2,
                capthick=0.3, elw=0.3, marker='s', ms=1.0, ninc=30, lw=1.0):
    """Script to make a matplotlib SUBplot of energy vs time.

    Parameters
    ----------
    list_of_results : list
        List of objects containing results as processed by the function
        load_all_information defined below.
    list_of_colors : list
        List of strings containing the corresponding colors of the lines.
    list_of_betas : list
        List of floats containing the corresponding values for beta, the
        inverse temperature.
    n_tracers : int
        Number of tracers, used for determining the standard error of the mean.
    title : str
        Title of the plot (optional, default is None).
    xscale : str
        Scale of the plot on the x axis, default for energy graph is 'log'.
    yscale : str
        Scale of the plot on the y axis, default for energy graph is None
        (meaning linear by default).
    DFS : int
        Default font size.
    capsize : float
        Size of the error bar cap sizes.
    capthick : float
        Size of the error bar cap thickness.
    elw : float
        Error bar line width.
    marker : str
        Marker style.
    ms : float
        Marker size.
    ninc : int
        Samples the grid at every ninc increments. Allows for clearer plots.
    lw : float
        Line width.
    """

    if title is not None:
        plt.title(r'%s' % title, fontsize=DFS)

    if xscale is not None:
        plt.xscale(xscale)

    if yscale is not None:
        plt.yscale(yscale)

    for ii in range(len(list_of_results)):
        plt.errorbar(list_of_results[ii]['tau_grid'][::ninc],
                     list_of_results[ii]['E'][::ninc],
                     np.array(list_of_results[ii]['dE'][::ninc]) /
                     np.sqrt(n_tracers - 1), linewidth=lw,
                     color=list_of_colors[ii], marker=marker, ms=ms,
                     capthick=capthick, capsize=capsize, elinewidth=elw,
                     label=r"$\beta=%.01f$" % list_of_betas[ii])


def handle_psi_counters(list_of_counters):
    """The counters need to be handled with some care, since they may be
    of different lengths. This function takes a list of counter classes."""

    # First, get the max length of this particular list of counters:
    maxL = max([len(x) for x in list_of_counters])

    # Create a numpy array of the appropriate size
    arr = np.zeros((len(list_of_counters), maxL))

    # Append to this array the *probabilities*, not the values themselves
    for ii, counter in enumerate(list_of_counters):  # Iterate over n_tracers

        # First, compute the total number of elements in the counter:
        s = sum([value for key, value in counter.items()])

        # Then append the probabilities to the array
        cc = 0
        for key, value in counter.items():
            arr[ii, cc] = value / s
            cc += 1

    # When done, return the mean and standard error in the mean of the
    # probabilities:
    _mean = np.mean(arr, axis=0).squeeze()
    _std_mean = np.std(arr, axis=0).squeeze()

    return _mean, _std_mean


def load_all_information(path, proto, string_index, n_tracers):
    """Loads in the tau grid, energies, minimum radius, psi and Pi values.

    Parameters
    ----------
    path : str
        The path to the directory containing the data.
    proto : int
        Protocol of the run.
    string_index : str
        The index which determines the parameter set, need to manually zfill.
    n_tracers : int
        The number of tracers, usually 200.
    """

    all_E = []
    all_S = []

    all_psiB = []
    all_psiC = []

    all_PiB = []
    all_PiC = []

    zfill_index = int(np.floor(np.log10(n_tracers))) + 1

    for ii in range(n_tracers):
        p = os.path.join(path, 'p%i-%s-%s.pkl'
                         % (proto, string_index, str(ii).zfill(zfill_index)))
        [tau_sample_grid, eAVG, eSTD, rminAVG, rminSTD, psiB, psiC,
         pi_grid_sample_1, Pi_basin_final_output, Pi_config_final_output] \
            = pickle.load(open(p, 'rb'))

        # Append the energies
        all_E.append(eAVG)
        all_S.append(eSTD)

        # Append the counters to be processed later
        all_psiB.append(psiB)
        all_psiC.append(psiC)

        # Append the Pi counters
        all_PiB.append(Pi_basin_final_output)
        all_PiC.append(Pi_config_final_output)

    # Complete processing of the energies vi error propagation
    all_E = np.array(all_E)
    all_S = np.array(all_S)
    avg_all_E = np.mean(all_E, axis=0).squeeze()
    avg_all_E_S = np.sqrt(np.mean(all_S**2, axis=0)).squeeze()

    # Similar procedure for the PiB and PiC
    all_PiB = np.array(all_PiB)
    all_PiC = np.array(all_PiC)
    avg_all_PiB = np.mean(all_PiB, axis=0).squeeze()
    avg_all_PiB_S = np.std(all_PiB, axis=0).squeeze()
    avg_all_PiC = np.mean(all_PiC, axis=0).squeeze()
    avg_all_PiC_S = np.std(all_PiC, axis=0).squeeze()

    # Finally manage the tricky counter classes
    avg_all_psiB, avg_all_psiB_S = handle_psi_counters(all_psiB)
    avg_all_psiC, avg_all_psiC_S = handle_psi_counters(all_psiC)

    d = {
        'tau_grid': tau_sample_grid,
        'E': avg_all_E,
        'dE': avg_all_E_S,
        'psiC': avg_all_psiC,
        'dpsiC': avg_all_psiC_S,
        'psiB': avg_all_psiB,
        'dpsiB': avg_all_psiB_S,
        'pi_grid': pi_grid_sample_1,
        'PiC': avg_all_PiC,
        'dPiC': avg_all_PiC_S,
        'PiB': avg_all_PiB,
        'dPiB': avg_all_PiB_S
    }

    return d
