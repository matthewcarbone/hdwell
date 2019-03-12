#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone & Marco Baity-Jesi"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"


import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
from cycler import cycler

from .aux import order_of_magnitude
from .templates import PLOTTING_INFO_TEMPLATE, PLOTTING_PROTOCOL_MAP

from . import logger  # noqa
lg = logging.getLogger(__name__)

# Custom color cycle.
mpl.rcParams['axes.prop_cycle'] = cycler('color', ['b', 'c', '#ffdb00',
                                                   'mediumseagreen', '#ffa904',
                                                   'pink', '#ee7b06',
                                                   '#a12424', '#400b0b', 'r'])

AVAILABLE_PLOTTING_PARAMETERS = [1]


def plot_actual(run_path, plotting_params, df):

    if plotting_params['protocol'] not in AVAILABLE_PLOTTING_PARAMETERS:
        raise RuntimeError("Unsupported plotting protocol %i"
                           % plotting_params['protocol'])

    p = PLOTTING_PROTOCOL_MAP[plotting_params['protocol']]
    dpi = plotting_params['dpi']

    # First step: read in all data corresponding to the four observables in
    # question: avg e, psi basin/config and the memory function.
    # TODO: memory function
    df = df.sort_values(by=[p['group_by'], p['to_plot']])

    if p['plot_maxes']:
        df = df.query('nmc == "%s" and nvec == "%s"'
                      % (str(np.max(df.nmc.unique())),
                         str(np.max(df.nvec.unique()))))

    groups = df[p['group_by']].unique()
    zfill_index = order_of_magnitude(1 + np.max(df['loc'].unique())) + 1

    # Clear past figures and initialize a new one.
    # Average energy ----------------------------------------------------------
    plt.clf()
    plt.figure(figsize=(8.5, 11))

    lg.info("Average Energy...")
    for ii, g in enumerate(groups):
        lg.info("%i // %i" % (ii, g))

        plt.gca().set_prop_cycle(None)  # Reset colormap
        ax = plt.subplot(len(groups), 1, ii + 1)

        # For each group, plot 'to_plot'.
        df_temp = df.query('%s == "%s"' % (p['group_by'], g))

        for index, row in df_temp.iterrows():
            # Load in the average energies.
            loc_str = str(row['loc']).zfill(zfill_index)
            energy_path = os.path.join(run_path, loc_str, 'avg_e.pkl')
            sample_e_path = os.path.join(run_path, loc_str, 'sample_e.pkl')
            avg_e = pickle.load(open(energy_path, 'rb'))
            sample_e = pickle.load(open(sample_e_path, 'rb'))
            if ii == 0:
                label = "%s" % row['beta']
            else:
                label = None
            if p['avg_e_scale'] == 'log':
                plt.semilogx(sample_e, avg_e, 'o', markersize=1, label=label)
            else:
                plt.plot(sample_e, avg_e, 'o', markersize=1, label=label)
            plt.ylabel(r"$\langle E \rangle$")

            if index == len(df_temp.index) - 1:
                plt.xlabel(r"$t$")

        plt.text(0.5, 0.1, "%s = %i" % (p['group_by'], g),
                 ha='center', va='top', transform=ax.transAxes, fontsize=14)

        if p['avg_e_best_fit'] and p['avg_e_scale'] == 'log':
            # This is not efficient. TODO: refactor.

            sample_e_log = np.log10(sample_e)
            last_n = p['last_n_points']

            plt.gca().set_prop_cycle(None)  # Reset colormap
            for index, row in df_temp.iterrows():
                loc_str = str(row['loc']).zfill(zfill_index)
                energy_path = os.path.join(run_path, loc_str, 'avg_e.pkl')
                sample_e_path = os.path.join(run_path, loc_str, 'sample_e.pkl')
                sample_e = pickle.load(open(sample_e_path, 'rb'))
                avg_e = pickle.load(open(energy_path, 'rb'))
                m = np.polyfit(sample_e_log[-last_n:], avg_e[-last_n:], deg=1)
                best_fit = np.polyval(m, sample_e_log)
                plt.semilogx(sample_e, best_fit, '--', alpha=0.5,
                             label="y = %.02f * x + %.02f" % (m[0], m[1]))

        plt.ylim(top=0.0)

        # Only need one legend.
        plt.legend(title="%s" % p['to_plot'], fontsize=6)

        if ii != len(groups) - 1:
            plt.tick_params(labelbottom=False)

    plt.subplots_adjust(hspace=0.05)

    plt.savefig(os.path.join(run_path, 'avg_e.pdf'), dpi=dpi,
                bbox_inches='tight')


def plotting_tool(data_path, plotting_params, prompt=True):
    """Plots all data available in the `DATA_hdwell` directory."""

    all_dirs = os.listdir(data_path)  # List all run directories.

    for directory in all_dirs:
        run_path = os.path.join(data_path, directory)
        param_path = os.path.join(run_path, 'params.csv')
        df = pd.read_csv(param_path)

        if prompt:
            x_ = PLOTTING_INFO_TEMPLATE.format(directory=run_path,
                                               beta=df.beta.unique(),
                                               dims=df.N.unique(),
                                               nmc=df.nmc.unique(),
                                               nvec=df.nvec.unique(),
                                               betac=df.lambda_prime.unique(),
                                               ptype=df.ptype.unique(),
                                               protocol=df.protocol.unique())

            if input(x_) != 'y':
                print("Skipping.")
                continue
            else:
                print("Plotting...")

        plot_actual(run_path, plotting_params, df)
