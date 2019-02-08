#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Marco Baity-Jesi"
__maintainer__ = "Matthew R. Carbone & Marco Baity-Jesi"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

"""Sets the default logging format for the console and LOG file. These
settings are fallbacks and are overwritten by the run.py file in the project
directory.

A summary of the logging protocol is as follows. The logging level may be set
for each handler. Here there are two: `fh` for the LOG file, and `ch` for the
console. Levels are set to DEBUG and WARNING, respectively, to begin, but are
overwritten in run.py depending on user input. The levels have the following
meaning.

DEBUG    : The lowest level used for outputting any and all information about
           the program. Only used when... debugging your program.
INFO     : General information. Usually also for debugging purposes since
           production runs should usually be accompanied by as little console
           output as possible (and that output should be piped to LOG).
WARNING  : Also generally not output to the console. Warnings should not
           compromise the execution, but are something to be taken note of so
           that files can be modified later.
ERROR    : Potential threat to the integrity of the run. These should always be
           piped to the console and should override any silent protocol.
CRITICAL : Surefire threat to the integrity of the run. The user should
           probably halt the job then and there to investigate.
"""


import logging


logging.getLogger().setLevel(logging.DEBUG)

logger_string_format = '%(asctime)s %(levelname)-8s' \
                       '[%(filename)s:%(lineno)d] %(message)s'

formatter = logging.Formatter(logger_string_format)

fh = logging.FileHandler('LOG')
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.WARNING)

logging.getLogger().addHandler(fh)
logging.getLogger().addHandler(ch)
