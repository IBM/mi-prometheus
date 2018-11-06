#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
grid_worker.py:

    - Contains the definition of the ``GridWorker`` class, base for all grid workers, such as ``GridTrainerCPU`` \
    & ``GridAnalyzer``. These grid workers do not inherit from ``Worker``, as they are different in behavior. \
    Rather, they reuse the base workers to manage grid of experiments (by calling them using the command lines).

    - This class also contains the definition of the default command line arguments of the grid workers.

"""
__author__ = "Vincent Marois & Tomasz Kornuta"

import os
import psutil

import logging
import argparse
from abc import abstractmethod

from miprometheus.utils.app_state import AppState
from miprometheus.utils.param_interface import ParamInterface


class GridWorker(object):
    """
    Base abstract class for the grid workers.
    All grid workers should subclass it and override the relevant methods.
    """

    def __init__(self, name="GridWorker", use_gpu=False):
        """
        Base constructor for all grid workers:

            - Initializes the AppState singleton:

                >>> self.app_state = AppState()

            - Initializes the Parameter Registry:

                >>> self.params = ParamInterface()

            - Defines the logger:

                >>> self.logger = logging.getLogger(name=self.name)

            - Creates parser and adds default worker command line arguments (you can display them with ``--h``).

        :param name: Name of the worker (DEFAULT: "GridWorker").
        :type name: str

        :param use_gpu: Indicates whether the worker should use GPU or not. Value coming from the subclasses \
         (e.g. ``GridTrainerCPU`` vs ``GridTrainerGPU``) (DEFAULT: False).
        :type use_gpu: bool

        """
        # Call base constructor.
        super(GridWorker, self).__init__()

        # Set worker name.
        self.name = name

        # Initialize the application state singleton.
        self.app_state = AppState()
        self.app_state.use_CUDA = use_gpu

        # Initialize parameter interface/registry.
        self.params = ParamInterface()

        # Load the default logger configuration.
        logger_config = {'version': 1,
                         'disable_existing_loggers': False,
                         'formatters': {
                             'simple': {
                                 'format': '[%(asctime)s] - %(levelname)s - %(name)s >>> %(message)s',
                                 'datefmt': '%Y-%m-%d %H:%M:%S'}},
                         'handlers': {
                             'console': {
                                 'class': 'logging.StreamHandler',
                                 'level': 'INFO',
                                 'formatter': 'simple',
                                 'stream': 'ext://sys.stdout'}},
                         'root': {'level': 'DEBUG',
                                  'handlers': ['console']}}

        logging.config.dictConfig(logger_config)

        # Create the Logger, set its label and logging level.
        self.logger = logging.getLogger(name=self.name)

        # Create parser with a list of runtime arguments.
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

        # Add arguments to the specific parser.
        # These arguments will be shared by all grid workers.
        self.parser.add_argument('--outdir',
                                 dest='outdir',
                                 type=str,
                                 default="./experiments",
                                 help='Path to the global output directory where the experiments folders '
                                      'will be / are stored. Affects all grid experiments.'
                                      ' (DEFAULT: ./experiments)')

        self.parser.add_argument('--savetag',
                                 dest='savetag',
                                 type=str,
                                 default='',
                                 help='Additional tag for the global output directory.')

        self.parser.add_argument('--ll',
                                 action='store',
                                 dest='log_level',
                                 type=str,
                                 default='INFO',
                                 choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                                 help="Log level for the experiments. (Default: INFO)")

        self.parser.add_argument('--li',
                                 dest='logging_interval',
                                 default=100,
                                 type=int,
                                 help='Statistics logging interval. Will impact logging to the logger and exporting to '
                                      'TensorBoard for the experiments. Do not affect the grid worker. '
                                      'Writing to the csv file is not impacted (interval of 1).'
                                      ' (Default: 100, i.e. logs every 100 episodes).')

        self.parser.add_argument('--agree',
                                 dest='confirm',
                                 action='store_true',
                                 help='Request user confirmation before starting the grid experiment.'
                                      '  (Default: False)')

    def setup_grid_experiment(self):
        """
        Setups the overall grid of experiments.

        Base method:

            - Parses command line arguments,

            - Sets the 3 default sections (training / validation / test) for the param registry, \
            sets seeds to unspecified and disable multiprocessing. Also saves the specified ``cuda`` key.

        .. note::

            Child classes should override this method, but still call its parent to draw the basic functionality \
            implemented here.

        """
        # Parse arguments.
        self.flags, self.unparsed = self.parser.parse_known_args()

        # Set logger depending on the settings.
        self.logger.setLevel(getattr(logging, self.flags.log_level.upper(), None))

        # add empty sections
        self.params.add_default_params({"training": {}})
        self.params.add_default_params({"validation": {}})
        self.params.add_default_params({"testing": {}})

        # set seeds to undefined (-1), pass CUDA value and deactivate multiprocessing for `DataLoader`.
        # It is important not to set the seeds here as they would be identical for all experiments.
        self.params["training"].add_default_params({"seed_numpy": -1,
                                                    "seed_torch": -1,
                                                    "dataloader": {'num_workers': 0}})

        self.params["validation"].add_default_params({"dataloader": {'num_workers': 0}})

        self.params["testing"].add_default_params({"seed_numpy": -1,
                                                   "seed_torch": -1,
                                                   "dataloader": {'num_workers': 0}})

    @abstractmethod
    def run_grid_experiment(self):
        """
        Main function of the ``GridWorker``, which essentially maps an experiment to available core or device.

        .. note::

            Abstract. Should be implemented in the subclasses.


        """


    def get_available_cpus(self):
        """ Function returns the number of available CPUs """
        # Check scheduler for number of available cpus - if OS offers that!
        if hasattr(os, 'sched_getaffinity'):
            return len(os.sched_getaffinity(0))

        proc = psutil.Process()
        # cpu_affinity() is only available on Linux, Windows and FreeBSD
        if hasattr(proc, 'cpu_affinity'):
            return len(proc.cpu_affinity())

        # Simply return CPU count :]
        return psutil.cpu_count()
