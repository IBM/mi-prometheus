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
grid_trainer_cpu.py:

    - This file contains the implementation of a worker spanning a grid of training experiments on \
     a collection of CPUs. It works by loading a template yaml file, modifying the resulting dict, and dumping \
      that as yaml into a temporary file. The specified ``Trainer`` is then executed using the temporary yaml \
      file as the task. This grid trainer will run as many concurrent jobs as possible.

"""
__author__ = "Alexis Asseman, Ryan McAvoy, Tomasz Kornuta, Vincent Marois"

import os
import yaml
import subprocess
from time import sleep
from datetime import datetime
from functools import partial
from tempfile import NamedTemporaryFile
from multiprocessing.pool import ThreadPool

from mip.workers.grid_worker import GridWorker


class GridTrainerCPU(GridWorker):
    """
    Grid Worker managing several training experiments on CPUs.

    Reuses a ``Trainer`` (can specify the ``classic`` one or the ``flexible`` one) to start one experiment.

    """

    def __init__(self, name="GridTrainerCPU", use_gpu=False):
        """
        Constructor for the ``GridTrainerCPU``:

            - Calls the base constructor to set the worker's name and add default command lines arguments,
            - Adds some ``GridTrainer`` specific command line arguments.

        :param name: Name of the worker (DEFAULT: "GridTrainerCPU").
        :type name: str

        :param use_gpu: Indicates whether the worker should use GPU or not.
        :type use_gpu: bool

        """
        # call base constructor
        super(GridTrainerCPU, self).__init__(name=name,use_gpu=use_gpu)

        # add one command line argument
        self.parser.add_argument('--config',
                                 dest='config',
                                 type=str,
                                 default='',
                                 help='Name of the configuration file(s) to be loaded. '
                                      'If specifying more than one file, they must be separated with coma ",".')

        self.parser.add_argument('--online_trainer',
                                 dest='online_trainer',
                                 action='store_true',
                                 help='Select the OnLineTrainer instead of the default OffLineTrainer.')

        self.parser.add_argument('--tensorboard',
                                 action='store',
                                 dest='tensorboard', choices=[0, 1, 2],
                                 type=int,
                                 help="If present, enable logging to TensorBoard. Available log levels:\n"
                                      "0: Log the collected statistics.\n"
                                      "1: Add the histograms of the model's biases & weights (Warning: Slow).\n"
                                      "2: Add the histograms of the model's biases & weights gradients "
                                      "(Warning: Even slower).")

    def setup_grid_experiment(self):
        """
        Setups a specific experiment.

        - Calls the ``super(self).setup_experiment()`` to parse arguments, sets the 3 default sections \
        (training / validation / test) and sets their dataloaders params.

        - Verifies that the specified config file is valid,

        - Parses it and recursively creates the configurations files for the grid tasks, overwriting \
        specific sections if indicated (`grid_overwrite` and/or `overwrite` (task specific),

        - Creates the output dir.


        """
        super(GridTrainerCPU, self).setup_grid_experiment()

        # Check if config file was selected.
        if self.flags.config == '':
            print('Please pass grid configuration file as --c parameter.')
            exit(-1)

        # Check if file exists.
        if not os.path.isfile(self.flags.config):
            print('Error: Grid configuration file {} does not exist.'.format(self.flags.config))
            exit(-2)

        try:  # open file and get parameter dictionary.
            with open(self.flags.config, 'r') as stream:
                grid_dict = yaml.safe_load(stream)

        except yaml.YAMLError as e:
            print("Error: Could not properly parse the {} grid configuration file.".format(self.flags.config))
            print('yaml.YAMLERROR:', e)
            exit(-3)

        # Get grid settings.
        try:
            experiment_repetitions = grid_dict['grid_settings']['experiment_repetitions']
            self.max_concurrent_run = grid_dict['grid_settings']['max_concurrent_runs']

        except KeyError:
            print("Error: The 'grid_settings' section must define 'experiment_repetitions' and 'max_concurrent_runs'.")
            exit(-4)

        # Check the presence of grid_overwrite section.
        if 'grid_overwrite' not in grid_dict:
            grid_overwrite_filename = None

        else:
            # Create temporary file with settings that will be overwritten for all tasks.
            grid_overwrite_file = NamedTemporaryFile(mode='w', delete=False)
            yaml.dump(grid_dict['grid_overwrite'], grid_overwrite_file, default_flow_style=False)
            grid_overwrite_filename = grid_overwrite_file.name

        # Check the presence of the tasks section.
        if 'grid_tasks' not in grid_dict:
            print("Error: Grid configuration is lacking the 'grid_tasks' section.")
            exit(-5)

        # Create temporary file
        param_interface_file = NamedTemporaryFile(mode='w', delete=False)
        yaml.dump(self.params.to_dict(), param_interface_file, default_flow_style=False)

        configs = []
        overwrite_files = []

        # Iterate through grid tasks.
        for task in grid_dict['grid_tasks']:

            try:
                # Retrieve the config(s).
                current_configs = param_interface_file.name + ',' + task['default_configs']

                # Extend them by grid_overwrite.
                if grid_overwrite_filename is not None:
                    current_configs = grid_overwrite_filename + ',' + current_configs

                if 'overwrite' in task:
                    # Create temporary file with settings that will be overwritten
                    # only for that particular task.
                    overwrite_files.append(NamedTemporaryFile(mode='w', delete=False))
                    yaml.dump(task['overwrite'], overwrite_files[-1], default_flow_style=False)
                    current_configs = overwrite_files[-1].name + ',' + current_configs

                # Get list of configs that need to be loaded.
                configs.append(current_configs)

            except KeyError:
                pass

        # at this point, configs should contains the str of config file(s) corresponding to the grid_tasks.

        # Create list of experiments
        self.experiments_list = []
        for _ in range(experiment_repetitions):
            self.experiments_list.extend(configs)

        self.logger.info('Number of experiments to run: {}'.format(len(self.experiments_list)))
        self.experiments_done = 0

        # create experiment directory label of the day
        self.outdir_str = self.flags.outdir + '_{0:%Y%m%d_%H%M%S}'.format(datetime.now())

        # add savetag
        if self.flags.savetag != '':
            self.outdir_str = self.outdir_str + "_" + self.flags.savetag + '/'

        # Prepare output paths for logging
        while True:  # Dirty fix: if log_dir already exists, wait for 1 second and try again
            try:
                os.makedirs(self.outdir_str, exist_ok=False)
            except FileExistsError:
                sleep(1)
            else:
                break

    def run_experiment(self, experiment_configs: str, prefix=""):
        """
        Setups the overall grid of experiments.

        :param experiment_configs: Configuration file(s) passed to the trainer using its `--c` argument. If indicating\
         several config files, they must be separated with coma ",".
        :type experiment_configs: str

        :param prefix: Prefix to position before the command string (e.g. 'cuda-gpupick -n 1'). Optional.
        :type prefix: str


        ..note::

            - Not using the ``--model`` argument of the ``Trainer`` to load a pretrained model.
            - Visualization is deactivated to avoid any user interaction.
            - Command-line arguments such as the logging interval (``--li``), tensorboard (``--t``) and log level \
            (``--ll``) are passed to the used ``Trainer``.


        """
        # set the command to be executed using the indicated Trainer
        if self.flags.online_trainer:
            command_str = "{}python3 workers/online_trainer.py".format(prefix)
        else:
            command_str = "{}python3 workers/offline_trainer.py".format(prefix)

        # Add gpu flag if required.
        if self.app_state.use_CUDA:
            command_str += " --gpu "

        # Add experiment config(s).
        command_str = command_str + " --c {0} --outdir " + self.outdir_str + ' --li ' + str(self.flags.logging_interval) \
                      + ' --ll ' + str(self.flags.log_level)
        command_str = command_str.format(experiment_configs)

        # Add tensorboard flag.
        if self.flags.tensorboard is not None:
            command_str += " --t " + str(self.flags.tensorboard)

        self.logger.info("Starting: {}".format(command_str))
        with open(os.devnull, 'w') as devnull:
            result = subprocess.run(command_str.split(" "), stdout=devnull)
        self.experiments_done += 1
        self.logger.info("Finished: {}".format(command_str))

        self.logger.info('Number of experiments done: {}/{}.'.format(self.experiments_done, len(self.experiments_list)))

        if result.returncode != 0:
            self.logger.info("Training exited with code: {}".format(result.returncode))

    def run_grid_experiment(self):
        """
        Main function of the ``GridTrainerCPU``.

        Maps the grid experiments to CPU cores in the limit of the maximum concurrent runs allowed or maximum\
         available cores.

        """
        # Ask for confirmation - optional.
        if self.flags.confirm:
            input('Press any key to continue')

        # Run in as many threads as there are CPUs available to the script
        max_processes = min(len(os.sched_getaffinity(0)), self.max_concurrent_run)

        with ThreadPool(processes=max_processes) as pool:
            func = partial(GridTrainerCPU.run_experiment, self, prefix="")
            pool.map(func, self.experiments_list)

        self.logger.info('Grid training experiments finished.')


def main():
    grid_trainer_cpu = GridTrainerCPU()

    # parse args, load configuration and create all required objects.
    grid_trainer_cpu.setup_grid_experiment()

    # GO!
    grid_trainer_cpu.run_grid_experiment()


if __name__ == '__main__':

    main()
