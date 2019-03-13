#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone & Marco Baity-Jesi"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

"""Templates and constant dictionaries."""

DANGER_ZONE_TEMPLATE = """
Warning: One more more danger zone run parameters is set to True. The
         parameter file indicates the following:

    * save_all_energies: {all_energies}
    * run_on_local:      {local}
    * pipe_all_to_LOG:   {log_all}

Do you still wish to continue? (y for yes)
"""

PLOTTING_INFO_TEMPLATE = """
Reading data from directory {directory}.

Parameter permutations:
    beta          {beta}
    N             {dims}
    nmc           {nmc}
    nvec          {nvec}
    beta_c        {betac}
    ptype         {ptype}
    protocol      {protocol}

Process this directory? (y for yes)
"""


"""The plotting protocol map defines pre-determined protocols for automatically
generating matplotlib plots. Note that some features, such as dpi, are defined
independently in the parameter file. The keys indicate the following:
    avg_e_scale : str
        Choices are 'log' or 'lin' for logarithmic or linear x-axis scale.
    last_n_points : int
        The number of points, starting from the end of the run, with which to
        use for the best fit line. Of course, like the above, ignored if
        avg_e_scale != 'log'.
    to_plot : str
        Which subset of the parameters to plot. Note that this can get out of
        hand very quickly, so this is limited to one entry.
    group_by : str
        Determines which sets of parameters get their own plots. For example,
        if 'group_by' is equal to 'N', then for every dimension given in the
        parameter file, that dimension gets its own plot, with the other
        desired parameters grouped into that plot. Essentially, `to_plot` and
        `group_by` completely determine what is plotted.
    plot_maxes : bool
        Plots only the combinations of the parameters nmc and nvec
        corresponding to the maximum combination.
"""

PLOTTING_PROTOCOL_MAP = {
    1: {
        'avg_e_scale': 'log',
        'last_n_points': 500,
        'to_plot': 'beta',
        'group_by': 'N',
        'plot_maxes': True
    }
}
