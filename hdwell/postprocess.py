#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone & Marco Baity-Jesi"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"


import os
import pickle
import pandas as pd
import numpy as np
import logging
from itertools import product
from tqdm import tqdm

from .aux import order_of_magnitude, makedir_if_not_exist

from . import logger  # noqa
lg = logging.getLogger(__name__)


def concat_loader(load_path):

    # Load in all energies:
    all_nrg = pickle.load(open(os.path.join(load_path, 'sas_all_e.pkl'), 'rb'))

    # Load in all min average radii:
    all_r = pickle.load(open(os.path.join(load_path,
                                          'sas_min_avg_r_e.pkl'), 'rb'))

    # Load in psi's:
    psi_basin = pickle.load(open(os.path.join(load_path,
                                              'psi_basin.pkl'), 'rb'))
    psi_config = pickle.load(open(os.path.join(load_path,
                                               'psi_config.pkl'), 'rb'))

    # Normalize psi_basin/config:
    norm_basin = np.sum(list(psi_basin.values()))
    for key, value in psi_basin.items():
        value /= norm_basin

    # Load in the memories: sas_memory_basin.pkl
    mem_basin = pickle.load(open(os.path.join(load_path,
                                              'sas_memory_basin.pkl'), 'rb'))
    mem_config = pickle.load(open(os.path.join(load_path,
                                               'sas_memory_config.pkl'), 'rb'))

    # Load in the rejection rates
    rates = pickle.load(open(os.path.join(load_path, 'rates.pkl'), 'rb'))

    return [all_nrg, all_r, psi_basin, psi_config, mem_basin, mem_config,
            rates]


def concatenate_psi(psi_b_list, psi_c_list):
    N = len(psi_b_list)
    np.testing.assert_equal(N, len(psi_c_list))
    psi_b_mat = np.zeros((N, len(psi_b_list[0])))
    psi_c_mat = np.zeros((N, len(psi_c_list[0])))

    for nn in range(N):
        bb = 0
        cl = psi_b_list[nn]
        cl_keys = list(cl.keys())
        cl_vals = list(cl.values())
        while bb < len(psi_b_list[nn]):
            try:
                psi_b_mat[nn, cl_keys[bb]] = cl_vals[bb]
                bb += 1
            except IndexError:
                psi_b_mat = \
                    np.concatenate((psi_b_mat, np.zeros((N, 1))), axis=-1)

    for nn in range(N):
        bb = 0
        cl = psi_c_list[nn]
        cl_keys = list(cl.keys())
        cl_vals = list(cl.values())
        while bb < len(psi_c_list[nn]):
            try:
                psi_c_mat[nn, cl_keys[bb]] = cl_vals[bb]
                bb += 1
            except IndexError:
                psi_c_mat = \
                    np.concatenate((psi_c_mat, np.zeros((N, 1))), axis=-1)

    return [psi_b_mat, psi_c_mat]


def concatenator(data_path, prompt=True, s_by='beta'):
    all_dirs = os.listdir(data_path)  # List all run directories.
    all_dirs.sort()

    for directory in all_dirs:
        run_path = os.path.join(data_path, directory)
        makedir_if_not_exist(os.path.join(run_path, 'concat'))

        if prompt:
            x_ = input("Concat: %s" % run_path)
            if x_ != 'y':
                continue

        param_path = os.path.join(run_path, 'params.csv')
        df = pd.read_csv(param_path, index_col=0)

        # Get all combinations of unique entries in the df.
        a = [df[name].unique() for name, values in df.iteritems()
             if name != 'loc' and name != 'nvec']
        col_headers = [name for name, values in df.iteritems()
                       if name != 'loc' and name != 'nvec']
        all_combos = list(product(*a))
        zf_index = order_of_magnitude(1 + np.max(df['loc'].unique())) + 1

        for jj, combo in tqdm(enumerate(all_combos)):
            query_string = ['%s == "%s"' % (col_headers[ii], combo[ii])
                            for ii in range(len(col_headers))]
            query_string = ' and '.join(query_string)
            sub_df = df.query(query_string)

            # Within each of these sub dataframes, concatenate
            index = 0
            if len(df['nvec'].unique()) != 1:
                raise RuntimeError("Non-unique nvec.")

            # Initialize some matrices as None
            e_mat, r_mat, mem_c_mat, mem_b_mat = None, None, None, None
            psi_b_list = []
            psi_c_list = []
            rej_rate = []
            up_rate = []
            down_rate = []
            outside_ball_rate = []

            for __, row in tqdm(sub_df.iterrows()):
                str_row = str(int(row['loc'])).zfill(zf_index)
                [e, r, psi_b, psi_c, mem_b, mem_c, rates] = \
                    concat_loader(os.path.join(run_path, str_row))
                psi_b_list.append(psi_b)
                psi_c_list.append(psi_c)
                rej_rate.append(rates[0])
                up_rate.append(rates[1])
                down_rate.append(rates[2])
                outside_ball_rate.append(rates[3])

                if index == 0:
                    e_mat = e
                    r_mat = r
                    mem_c_mat = mem_c
                    mem_b_mat = mem_b

                    # Load in the energy/min_radius grids:
                    st = os.path.join(run_path, str_row, 'sample_e.pkl')
                    e_grid = pickle.load(open(st, 'rb'))

                    # Load in the memory grids:
                    st = os.path.join(run_path, str_row, 'memory_basin.pkl')
                    [__, mem_grid, domega] = pickle.load(open(st, 'rb'))
                    first_str_row = str_row

                else:
                    e_mat = np.concatenate((e_mat, e), axis=1)
                    r_mat = np.concatenate((r_mat, r), axis=1)
                    mem_c_mat = np.concatenate((mem_c_mat, mem_c), axis=1)
                    mem_b_mat = np.concatenate((mem_b_mat, mem_b), axis=1)

                index += 1

            [psi_b_mat, psi_c_mat] = concatenate_psi(psi_b_list, psi_c_list)

            concat_loc_path = os.path.join(run_path, 'concat', first_str_row)
            makedir_if_not_exist(concat_loc_path)

            # Save the energy:
            average_energy = np.mean(e_mat, axis=1)
            std_energy = np.std(e_mat, axis=1)
            e_path = os.path.join(concat_loc_path, 'energy.pkl')
            pickle.dump([e_grid, average_energy, std_energy],
                        open(e_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

            # Save the min radius:
            average_min_r = np.mean(r_mat, axis=1)
            std_min_r = np.std(r_mat, axis=1)
            r_path = os.path.join(concat_loc_path, 'min_radius.pkl')
            pickle.dump([e_grid, average_min_r, std_min_r],
                        open(r_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

            # Memory functions
            average_mem_b = np.mean(mem_b_mat, axis=1)
            std_mem_b = np.std(mem_b_mat, axis=1)
            mem_b_path = os.path.join(concat_loc_path, 'mem_b.pkl')
            pickle.dump([mem_grid, average_mem_b, std_mem_b],
                        open(mem_b_path, 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)

            average_mem_c = np.mean(mem_c_mat, axis=1)
            std_mem_c = np.std(mem_c_mat, axis=1)
            mem_c_path = os.path.join(concat_loc_path, 'mem_c.pkl')
            pickle.dump([mem_grid, average_mem_c, std_mem_c],
                        open(mem_c_path, 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)

            # Psi functions
            average_psi_b = np.mean(psi_b_mat, axis=0)
            std_psi_b = np.std(psi_b_mat, axis=0)
            psi_b_path = os.path.join(concat_loc_path, 'psi_b.pkl')
            pickle.dump([average_psi_b, std_psi_b],
                        open(psi_b_path, 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)

            average_psi_c = np.mean(psi_c_mat, axis=0)
            std_psi_c = np.std(psi_c_mat, axis=0)
            psi_c_path = os.path.join(concat_loc_path, 'psi_c.pkl')
            pickle.dump([average_psi_c, std_psi_c],
                        open(psi_c_path, 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)

            rej_path = os.path.join(concat_loc_path, 'rej.pkl')
            pickle.dump([rej_rate, up_rate, down_rate, outside_ball_rate],
                        open(rej_path, 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
