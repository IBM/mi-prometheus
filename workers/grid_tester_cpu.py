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
import argparse
import subprocess
from functools import partial
from multiprocessing.pool import ThreadPool

import workers.worker as worker
from workers.worker import Worker


def add_arguments(parser: argparse.ArgumentParser):
    """
    Add arguments to the specific parser.

    These arguments are related to the  ``GridTesterCPU``.

    :param parser: ``argparse.ArgumentParser``
    """
    # add here all arguments used by the GridTesterCPU.
    parser.add_argument('--outdir',
                        dest='outdir',
                        type=str,
                        default="./experiments",
                        help='Path to output directory where the experiments are stored (DEFAULT: ./experiments)')

    # get number_of_repetitions
    parser.add_argument('--n',
                        dest='num_tests',
                        type=int,
                        default=1,
                        help='Number of test experiments to run for each model.')


class GridTesterCPU(Worker):
    """
    Implementation of the Grid Tester running on CPUs.

    Reuses the ``Tester`` to start one test experiment.

    """

    def __init__(self, flags: argparse.Namespace):
        """
        Constructor for the ``GridTesterCPU``:

            - constructs all possible sub-directories paths in `flags.outdir`
            - only keeps the ones that both `validation.csv` and `training.csv`


        :param flags: Parsed arguments from the parser.

        """
        self.name = 'GridTesterCPU'

        # call base constructor
        super(GridTesterCPU, self).__init__(flags)

        directory_chckpnts = flags.outdir
        num_tests = flags.num_tests

        # get all sub-directories paths in outdir, repeating according to flags.num
        self.experiments_list = []

        for _ in range(num_tests):
            for root, dirs, files in os.walk(directory_chckpnts, topdown=True):
                for name in dirs:
                    self.experiments_list.append(os.path.join(root, name))

        # Keep only the folders that contain validation.csv and training.csv TODO: Why?
        self.experiments_list = [elem for elem in self.experiments_list if os.path.isfile(
            elem + '/validation.csv') and os.path.isfile(elem + '/training.csv')]

        # check if the files are empty except for the first line TODO: Why?
        self.experiments_list = [elem for elem in self.experiments_list if os.stat(
            elem + '/validation.csv').st_size > 24 and os.stat(elem + '/training.csv').st_size > 24]

    def run_experiment(self, experiment_path: str, prefix=""):
        """
        Runs a test on the specified model (experiment_path) using the ``Tester``.

        :param experiment_path: Path to an experiment folder containing a trained model.
        :type experiment_path: str

        :param prefix: Prefix to position before the command string (e.g. 'cuda-gpupick -n 1'). Optional.
        :type prefix: str

        ..note::

            Visualization is deactivated to avoid any user interaction.


        """
        path_to_model = os.path.join(experiment_path, 'models/model_best.pt')

        # check if models list is empty
        if not os.path.isfile(path_to_model):
            self.logger.warning('The indicated model {} does not exist on file.'.format(path_to_model))

        else:

            # Run the test
            command_str = "{}python3 workers/tester.py --model {}".format(prefix, path_to_model)

            self.logger.info("Starting: {}".format(command_str))
            with open(os.devnull, 'w') as devnull:
                result = subprocess.run(command_str.split(" "), stdout=devnull)
            self.logger.info("Finished: {}".format(command_str))

            if result.returncode != 0:
                self.logger.info("Testing exited with code: {}".format(result.returncode))

    def forward(self, flags: argparse.Namespace):
        """
        Main function of the ``GridTesterCPU``.

        Maps the grid experiments to CPU cores in the limit of the maximum concurrent runs allowed or maximum\
         available cores.

        :param flags: Parsed arguments from the parser.

        """
        # Ask for confirmation - optional.
        if flags.confirm:
            input('Press any key to continue')

        # Run in as many threads as there are CPUs available to the script
        with ThreadPool(processes=len(os.sched_getaffinity(0))) as pool:
            func = partial(GridTesterCPU.run_experiment, self, prefix="")
            pool.map(func, self.experiments_list)

        self.logger.info('Grid test experiments finished.')


if __name__ == '__main__':
    # Create parser with list of  runtime arguments.
    argp = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # add default arguments
    worker.add_arguments(argp)

    # add grid trainers-specific arguments
    add_arguments(argp)

    # Parse arguments.
    FLAGS, unparsed = argp.parse_known_args()

    grid_tester_cpu = GridTesterCPU(FLAGS)
    grid_tester_cpu.forward(FLAGS)
