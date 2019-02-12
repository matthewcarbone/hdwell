#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone & Marco Baity-Jesi"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

"""Execution file for the hdwell project. Details to follow."""

import numpy as np
import argparse
import logging
import os
import matplotlib.pyplot as plt

from hdwell import logger
from hdwell.test_simulation import test_metropolis
lg = logging.getLogger(__name__)


PROTOCOL_CHOICES = ['test']


def parser():
    """Uses argparse to parse command line arguments and returns said
    arguments."""

    ap = argparse.ArgumentParser(allow_abbrev=False)

    ap.add_argument('--debug', action='store_true', dest='debug',
                    default=False, help='run in debug mode')
    ap.add_argument('--info', action='store_true', dest='info',
                    default=False, help='run in info mode')
    ap.add_argument('--nolog', action='store_true', dest='nolog',
                    default=False, help='force LOG file output to warning '
                                        'level')
    ap.add_argument('-p', '--protocol', dest='protocol',
                    choices=PROTOCOL_CHOICES,
                    help='set the protocol for the run')

    return ap.parse_args()


def set_logging_level(args):
    """Sets the logging level to one of the potential three described in the
    module docstring: normal, info or debug. Returns the bool value for
    `silent` which can be used to explicitly silence things like the tqdm
    progress bar."""

    if args.debug and args.info:
        raise RuntimeError("Cannot run in both debug and info mode "
                           "simultaneously.")

    silent = True

    if args.debug and not args.info:
        logger.fh.setLevel(logging.DEBUG)
        logger.ch.setLevel(logging.DEBUG)

    elif not args.debug and args.info:
        logger.fh.setLevel(logging.DEBUG)
        logger.ch.setLevel(logging.INFO)

    else:
        logger.fh.setLevel(logging.INFO)
        logger.ch.setLevel(logging.ERROR)
        silent = False

    if args.nolog:
        logger.fh.setLevel(logging.WARNING)

    return silent


if __name__ == '__main__':
    args = parser()
    silent = set_logging_level(args)

    if args.protocol == 'test':
        [x1, e1] = test_metropolis(10000, N=1, dim=3)
        [x2, e2] = test_metropolis(10000, N=1, dim=30)
        [x3, e3] = test_metropolis(10000, N=1, dim=300)
        [x4, e4] = test_metropolis(10000, N=1, dim=3000)
        [x5, e5] = test_metropolis(10000, N=1, dim=30000)
        nn = len(e1)

        working_directory = os.getcwd()

        plt.plot(np.linspace(1, nn, nn), e1, 'k', label="3")
        plt.plot(np.linspace(1, nn, nn), e2, 'g', label="30")
        plt.plot(np.linspace(1, nn, nn), e3, 'b', label="300")
        plt.plot(np.linspace(1, nn, nn), e4, 'r', label="3000")
        plt.plot(np.linspace(1, nn, nn), e5, 'c', label="30000")
        plt.ylabel("Energy")
        plt.xlabel("Monte Carlo Time")
        plt.savefig(os.path.join(working_directory, 'test.pdf'), dpi=300,
                    bbox_inches='tight')
