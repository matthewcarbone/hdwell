#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone & Marco Baity-Jesi"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

"""Templates"""

DANGER_ZONE_TEMPLATE = """
Warning: One more more danger zone run parameters is set to True. The
         parameter file indicates the following:

    * save_all_energies: {all_energies}
    * run_on_local:      {local}
    * pipe_all_to_LOG:   {log_all}

Do you still wish to continue? (y for yes)\n
"""
