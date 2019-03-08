#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone & Marco Baity-Jesi"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

"""Execution file for the hdwell project. The general workflow of this code
goes as follows.
    1. Command line arguments and yaml paramter file are read in here and
       passed to the `run_all` function. Note that parameter files may be
       overwritten by command line arguments.
    2. `run_all` (part of execute.py) summarizes the details of the current
       run and will prompt the user (unless --noprompt flag) to continue with
       the specified execution protocol. If the user agrees to continue, then
       `run_all` will pass to the particular function for that protocol.
    3. Specific protocol functions e.g. `execute_protocol_1` parse the
       parameters as necessary and (if necessary) provide final user prompts.
       After the user agrees to continue, it runs bash scripts via the
       subprocess library, submitting the jobs to the SLURM job controller, or
       local machine if specified.
    4. The sub.py file contains scripts that are *meant to be called directly*
       by other functions in this package and run on the compute nodes. The
       bash/sbatch scripts will always call sub.py directly.
"""

import argparse
import logging
import yaml
import os

from hdwell import logger
from hdwell.execute import run_all

lg = logging.getLogger(__name__)

WORKDIR = os.getcwd()
HOMEDIR = os.path.expanduser("~")
PROTOCOL_CHOICES = ['actual']


def get_target_dir(directory_override):
    if directory_override is None:
        return HOMEDIR
    elif directory_override == 'wd':
        return WORKDIR
    else:
        return directory_override


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
    ap.add_argument('--noprompt', action='store_false', dest='prompt',
                    default=True, help='ignore prompts')

    ap.add_argument('-p', '--protocol', dest='protocol',
                    choices=PROTOCOL_CHOICES, default='actual',
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

    if args.debug and not args.info:
        logger.fh.setLevel(logging.DEBUG)
        logger.ch.setLevel(logging.DEBUG)

    elif not args.debug and args.info:
        logger.fh.setLevel(logging.DEBUG)
        logger.ch.setLevel(logging.INFO)

    else:
        logger.fh.setLevel(logging.INFO)
        logger.ch.setLevel(logging.ERROR)

    if args.nolog:
        logger.fh.setLevel(logging.WARNING)


if __name__ == '__main__':
    args = parser()
    set_logging_level(args)

    params = yaml.safe_load(open(os.path.join(WORKDIR, "params.yaml")))
    target_directory = get_target_dir(params['directory_override'])
    run_all(params, target_directory, prompt=args.prompt)
