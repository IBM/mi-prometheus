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
import shutil
import subprocess
from functools import partial
from multiprocessing.pool import ThreadPool

from miprometheus.grid_workers.grid_worker import GridWorker


class GridTesterCPU(GridWorker):
    """
    Implementation of the Grid Tester running on CPUs.

    Reuses the ``Tester`` to start one test experiment.

    """

    def __init__(self, name="GridTesterCPU", use_gpu=False):
        """
        Constructor for the ``GridTesterCPU``:

            - Calls the base constructor to set the worker's name and add default command lines arguments,
            - Adds some ``GridTrainer`` specific command line arguments.

        :param name: Name of the worker (DEFAULT: "GridTesterCPU").
        :type name: str

        :param use_gpu: Indicates whether the worker should use GPU or not.
        :type use_gpu: bool

        """
        # call base constructor
        super(GridTesterCPU, self).__init__(name=name,use_gpu=use_gpu)

        # Get number_of_repetitions
        self.parser.add_argument('--repeat',
                                 dest='experiment_repetitions',
                                 type=int,
                                 default=1,
                                 help='Number of experiment repetitions to run for each model'
                                 ' (DEFAULT=1)')

        # Get number_of_repetitions
        self.parser.add_argument('--max_concur_runs',
                                 dest='max_concurrent_runs',
                                 type=int,
                                 default=-1,
                                 help='Value limiting the number of concurently running experiments.'
                                    'The set limit will be truncated by number of available CPUs/GPUs.'
                                    ' (DEFAULT=-1, meaning that it will be set to the number of CPUs/GPUs)')


    def setup_grid_experiment(self):
        """
         Setups the overall grid of experiments:

        - Calls the ``super(self).setup_experiment()`` to parse arguments,

        - Recursively creates the paths to the experiments folders, verifying that they are valid (e.g. \
        contain `validation_statistics.csv` and `training_statistics.csv`).


        :param cuda: Whether to use cuda or not. Default to ``False``.
        :type cuda: bool

        """
        super(GridTesterCPU, self).setup_grid_experiment()

        # Check the presence of mip-tester script.
        if shutil.which('mip-tester') is None:
            self.logger.error("Cannot localize the 'mip-tester' script! (hint: please use setup.py to install it)")
            exit(-1)

        self.experiment_rootdir = self.flags.expdir
        # Get grid settings.
        experiment_repetitions = self.flags.experiment_repetitions
        self.max_concurrent_runs = self.flags.max_concurrent_runs

        # get all sub-directories paths in expdir, repeating according to flags.num
        self.experiments_list = []

        for _ in range(experiment_repetitions):
            for root, dirs, _ in os.walk(self.experiment_rootdir, topdown=True):
                for name in dirs:
                    self.experiments_list.append(os.path.join(root, name))

        # Keep only the folders that contain best_model.pt in model subdirectory.
        # We assume that training configuration is there as well.
        self.experiments_list = [elem for elem in self.experiments_list 
            if os.path.isfile(elem + '/model_best.pt') ]

        # Check if there are some valid folders.
        if len(self.experiments_list) == 0:
            self.logger.error("There are no models in {} directory!".format(self.experiment_rootdir))
            exit(-2)

        # List folders.
        exp_str = "Found the following models in {} directory:\n".format(self.experiment_rootdir)
        exp_str += '='*80 + '\n'
        for exp in self.experiments_list:
            exp_str += " - {}/model_best.pt\n".format(exp)
        exp_str += '='*80 + '\n'
        self.logger.info(exp_str)

        self.logger.info('Number of experiments to run: {}'.format(len(self.experiments_list)))
        self.experiments_done = 0

        # Ask for confirmation - optional.
        if self.flags.user_confirm:
            try:
                input('Press <Enter> to confirm and start the grid of experiments\n')
            except KeyboardInterrupt:
                exit(0)            


    def run_grid_experiment(self):
        """
        Main function of the ``GridTesterCPU``.

        Maps the grid experiments to CPU cores in the limit of the maximum concurrent runs allowed or maximum\
         available cores.

        """
        try:

            # Check max number of child processes. 
            if self.max_concurrent_runs <= 0: # We need at least one proces!
                max_processes = self.get_available_cpus()
            else:    
                # Take into account the minimum value.
                max_processes = min(self.get_available_cpus(), self.max_concurrent_runs)
            self.logger.info('Spanning experiments using {} CPU(s) concurrently'.format(max_processes))

            # Run in as many threads as there are CPUs available to the script.
            with ThreadPool(processes=max_processes) as pool:
                func = partial(GridTesterCPU.run_experiment, self, prefix="")
                pool.map(func, self.experiments_list)

            self.logger.info('Grid testing finished')

        except KeyboardInterrupt:
            self.logger.info('Grid testing interrupted!')


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
        try:

            path_to_model = os.path.join(experiment_path, 'model_best.pt')
            self.logger.warning(path_to_model)

            # Run the test
            command_str = "{}mip-tester --model {} --li {} --ll {}".format(
                prefix, path_to_model,
                self.flags.logging_interval,
                self.flags.log_level)

            # Add gpu flag if required.
            if self.app_state.use_CUDA:
                command_str += " --gpu "

            self.logger.info("Starting: {}".format(command_str))
            with open(os.devnull, 'w') as devnull:
                result = subprocess.run(command_str.split(" "), stdout=devnull)
            self.experiments_done += 1
            self.logger.info("Finished: {}".format(command_str))

            self.logger.info(
                'Number of experiments done: {}/{}.'.format(self.experiments_done, len(self.experiments_list)))

            if result.returncode != 0:
                self.logger.info("Testing exited with code: {}".format(result.returncode))

        except KeyboardInterrupt:
            self.logger.info('Grid testing interrupted!')


def main():
    """
    Entry point function for the ``GridTesterCPU``.

    """
    grid_tester_cpu = GridTesterCPU()

    # parse args, load configuration and create all required objects.
    grid_tester_cpu.setup_grid_experiment()

    # GO!
    grid_tester_cpu.run_grid_experiment()


if __name__ == '__main__':

    main()
