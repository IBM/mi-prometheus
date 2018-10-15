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
grid_tester_cpu.py:

    - This file contains the implementation of a worker running the ``Tester`` on the results of a ``GridTrainer``
    using CPUs.

    - The input is a list of directories for each problem/model e.g. `experiments/serial_recall/dnc`, \
      and executes on every run of the model in that directory.

"""
__author__ = "Tomasz Kornuta & Vincent Marois"

import os
import subprocess
from functools import partial
from multiprocessing.pool import ThreadPool

from workers.grid_worker import GridWorker


class GridTesterCPU(GridWorker):
    """
    Implementation of the Grid Tester running on CPUs.

    Reuses the ``Tester`` to start one test experiment.

    """

    def __init__(self, name="GridTesterCPU"):
        """
        Constructor for the ``GridTesterCPU``:

            - Calls the base constructor to set the worker's name and add default command lines arguments,
            - Adds some ``GridTrainer`` specific command line arguments.

        :param name: Name of the worker (DEFAULT: "GridTesterCPU").
        :type name: str

        """
        # call base constructor
        super(GridTesterCPU, self).__init__(name=name)

        # add here all arguments used by the GridTesterCPU.
        self.parser.add_argument('--model',
                                 type=str,
                                 default='',
                                 dest='model',
                                 help='Path to the file containing the saved parameters'
                                      ' of the model to load (model checkpoint, should end with a .pt extension.)')

        # get number_of_repetitions
        self.parser.add_argument('--n',
                                 dest='num_tests',
                                 type=int,
                                 default=1,
                                 help='Number of test experiments to run for each model.')

    def setup_grid_experiment(self, cuda=False):
        """
         Setups the overall grid of experiments:

        - Calls the ``super(self).setup_experiment()`` to parse arguments,

        - Recursively creates the paths to the experiments folders, verifying that they are valid (e.g. \
        contain `validation_statistics.csv` and `training_statistics.csv`).


        :param cuda: Whether to use cuda or not. Default to ``False``.
        :type cuda: bool

        """
        super(GridTesterCPU, self).setup_grid_experiment(cuda=cuda)

        directory_chckpnts = self.flags.outdir
        num_tests = self.flags.num_tests

        # get all sub-directories paths in outdir, repeating according to flags.num
        self.experiments_list = []

        for _ in range(num_tests):
            for root, dirs, files in os.walk(directory_chckpnts, topdown=True):
                for name in dirs:
                    self.experiments_list.append(os.path.join(root, name))

        # Keep only the folders that contain validation.csv and training.csv
        self.experiments_list = [elem for elem in self.experiments_list if os.path.isfile(
            elem + '/validation_statistics.csv') and os.path.isfile(elem + '/training_statistics.csv')]

        # check if the files are not empty
        self.experiments_list = [elem for elem in self.experiments_list if os.stat(
            elem + '/validation_statistics.csv').st_size > 24 and os.stat(elem + '/training.csv_statistics').st_size > 24]

        self.logger.info('Number of experiments to run: {}'.format(len(self.experiments_list)))
        self.experiments_done = 0

    def run_experiment(self, experiment_path: str, prefix=""):
        """
        Runs a test on the specified model (experiment_path) using the ``Tester``.

        :param experiment_path: Path to an experiment folder containing a trained model.
        :type experiment_path: str

        :param prefix: Prefix to position before the command string (e.g. 'cuda-gpupick -n 1'). Optional.
        :type prefix: str

        ..note::

            - Visualization is deactivated to avoid any user interaction.
            - Command-line arguments such as the logging interval (``--li``) and log level (``--ll``) are passed \
             to the used ``Trainer``.

        """
        path_to_model = os.path.join(experiment_path, 'models/model_best.pt')

        # check if models list is empty
        if not os.path.isfile(path_to_model):
            self.logger.warning('The indicated model {} does not exist on file.'.format(path_to_model))

        else:

            # Run the test
            command_str = "{}python3 workers/tester.py --model {} --li {} --ll".format(prefix, path_to_model,
                                                                                       self.flags.logging_interval,
                                                                                       self.flags.log_level)

            self.logger.info("Starting: {}".format(command_str))
            with open(os.devnull, 'w') as devnull:
                result = subprocess.run(command_str.split(" "), stdout=devnull)
            self.experiments_done += 1
            self.logger.info("Finished: {}".format(command_str))
            print()
            self.logger.info(
                'Number of experiments done: {}/{}.'.format(self.experiments_done, len(self.experiments_list)))

            if result.returncode != 0:
                self.logger.info("Testing exited with code: {}".format(result.returncode))

    def run_grid_experiment(self):
        """
        Main function of the ``GridTesterCPU``.

        Maps the grid experiments to CPU cores in the limit of the maximum concurrent runs allowed or maximum\
         available cores.

        """
        # Ask for confirmation - optional.
        if self.flags.confirm:
            input('Press any key to continue')

        # Run in as many threads as there are CPUs available to the script
        with ThreadPool(processes=len(os.sched_getaffinity(0))) as pool:
            func = partial(GridTesterCPU.run_experiment, self, prefix="")
            pool.map(func, self.experiments_list)

        self.logger.info('Grid test experiments finished.')


if __name__ == '__main__':
    grid_tester_cpu = GridTesterCPU()

    # parse args, load configuration and create all required objects.
    grid_tester_cpu.setup_grid_experiment(cuda=False)

    # GO!
    grid_tester_cpu.run_grid_experiment()
