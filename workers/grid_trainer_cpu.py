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

    - This file contains the implementation of a worker spanning a grid of training experiments on\
     a collection of CPUs. It works by loading a template yaml file, modifying the resulting dict, and dumping\
      that as yaml into a temporary file. The ``Trainer`` is then executed using the temporary yaml file as the task.\
      It will run as many concurrent jobs as possible.

"""
__author__ = "Alexis Asseman, Ryan McAvoy, Tomasz Kornuta, Vincent Marois"


import os
import yaml
import argparse
import subprocess
from time import sleep
from datetime import datetime
from functools import partial
from tempfile import NamedTemporaryFile
from multiprocessing.pool import ThreadPool

import workers.worker as worker
from workers.worker import Worker


def add_arguments(parser: argparse.ArgumentParser):
    """
    Add arguments to the specific parser.

    These arguments are related to the  ``GridTrainerCPU``.

    :param parser: ``argparse.ArgumentParser``
    """
    # add here all arguments used by the GridTrainerCPU.
    parser.add_argument('--config',
                        dest='config',
                        type=str,
                        default='',
                        help='Name of the grid configuration file to be loaded')

    parser.add_argument('--episodic_trainer',
                        dest='episodic_trainer',
                        action='store_true',
                        help='Select the episodic Trainer instead of the default (epoch-based)'
                             ' Trainer. Useful for algorithmic tasks.')


class GridTrainerCPU(Worker):
    """
    Grid Worker managing several training experiments on CPUs.

    Reuses the ``Trainer`` (can specify the base one or the episodic one) to start one experiment.

    """

    def __init__(self, flags: argparse.Namespace, cuda=False):
        """
        Constructor for the ``GridTrainerCPU``:

            - Recursively loads the configuration files needed for the specified grid tasks,
            - Set up the log directory path.


        :param flags: Parsed arguments from the parser.

        :param cuda: Whether or not to use CUDA (cf ``GridTrainerGPU``). Default to False.
        :type cuda: bool


        """
        self.name = 'GridTrainerCPU'

        # call base constructor
        super(GridTrainerCPU, self).__init__(flags)

        # Check if config file was selected.
        if flags.config == '':
            print('Please pass grid configuration file as --c parameter')
            exit(-1)

        # Check if file exists.
        if not os.path.isfile(flags.config):
            print('Error: Grid configuration file {} does not exist'.format(flags.config))
            exit(-2)

        try:  # open file and get parameter dictionary.
            with open(flags.config, 'r') as stream:
                grid_dict = yaml.safe_load(stream)

        except yaml.YAMLError as e:
            print("Error: Coudn't properly parse the {} grid configuration file".format(flags.config))
            print('yaml.YAMLERROR:', e)
            exit(-3)

        # Get grid settings.
        try:
            experiment_repetitions = grid_dict['grid_settings']['experiment_repetitions']
            self.max_concurrent_run = grid_dict['grid_settings']['max_concurrent_runs']

        except KeyError:
            print("Error: The 'grid_settings' section must define 'experiment_repetitions' and 'max_concurrent_runs'")
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
            print("Error: Grid configuration is lacking the 'grid_tasks' section")
            exit(-5)

        # Create a configuration specific to this grid trainer:
        # set seeds to undefined (-1), CUDA to false and deactivate multiprocessing for `DataLoader`.
        # It is important not to set the seeds here as they would be identical for all experiments.

        self.param_interface["training"].add_custom_params({"seed_numpy": -1,
                                                            "seed_torch": -1,
                                                            "cuda": cuda,
                                                            "dataloader": {'num_workers': 0}})
        self.param_interface["validation"].add_custom_params({"dataloader": {'num_workers': 0}})

        # also doing it for GridTesters as they do not pass their ParamInterface to testers (it is reloaded
        # from training_configuration.yaml)
        self.param_interface["testing"].add_custom_params({"dataloader": {'num_workers': 0}})

        # Create temporary file
        param_interface_file = NamedTemporaryFile(mode='w', delete=False)
        yaml.dump(self.param_interface.to_dict(), param_interface_file, default_flow_style=False)

        configs = []
        overwrite_files = []

        # Iterate through grid tasks.
        for task in grid_dict['grid_tasks']:

            try:
                # Retrieve the config(s).
                current_configs = param_interface_file.name + ',' + task['default_configs']

                # Extend them by batch_overwrite.
                if grid_overwrite_filename is not None:
                    current_configs = grid_overwrite_filename + ',' + current_configs

                if 'overwrite' in task:
                    # Create temporary file with settings that will be overwritten
                    # only for that particular task.
                    overwrite_files.append(NamedTemporaryFile(mode='w', delete=False))
                    yaml.dump(task['overwrite'], overwrite_files[-1], default_flow_style=False)
                    current_configs = overwrite_files[-1].name + ',' + current_configs

                # Get list of configs that ne   ed to be loaded.
                configs.append(current_configs)

            except KeyError:
                pass

        # at this point, configs should contains the str of config file(s) corresponding to the grid_tasks

        # Create list of experiments
        self.experiments_list = []
        for _ in range(experiment_repetitions):
            self.experiments_list.extend(configs)

        # create experiment directory label of the day
        self.outdir_str = './experiments_' + '{0:%Y%m%d_%H%M%S}'.format(datetime.now())

        # add savetag
        if flags.savetag != '':
            self.outdir_str = self.outdir_str + "_" + flags.savetag + '/'

        # Prepare output paths for logging
        while True:  # Dirty fix: if log_dir already exists, wait for 1 second and try again
            try:
                os.makedirs(self.outdir_str, exist_ok=False)
            except FileExistsError:
                sleep(1)
            else:
                break

    def run_experiment(self, episodic_trainer, output_dir: str, experiment_configs: str, prefix=""):
        """
        Runs the specified experiment using one Trainer.

        :param episodic_trainer: Whether to use the EpisodicTrainer instead of the default Trainer
        :type episodic_trainer: bool

        :param output_dir: Output directory for the experiment files (logging, best model etc.)
        :type output_dir: str

        :param experiment_configs: Configuration file(s) passed to the trainer using its `--c` argument. If indicating\
        several config files, they must be separated with coma ",".
        :type experiment_configs: str

        :param prefix: Prefix to position before the command string (e.g. 'cuda-gpupick -n 1'). Optional.
        :type prefix: str


        ..note::

            Statistics exporting to TensorBoard is currently not activated.

            Visualization is deactivated to avoid any user interaction.


        """
        # set the command to be executed using the indicated Trainer
        if episodic_trainer:
            command_str = "{}python3 workers/episodic_trainer.py".format(prefix)
        else:
            command_str = "{}python3 workers/trainer.py".format(prefix)

        # add experiment config
        command_str = command_str + " --c {0} --outdir " + output_dir
        command_str = command_str.format(experiment_configs)

        self.logger.info("Starting: {}".format(command_str))
        with open(os.devnull, 'w') as devnull:
            result = subprocess.run(command_str.split(" "), stdout=devnull)
        self.logger.info("Finished: {}".format(command_str))

        if result.returncode != 0:
            self.logger.info("Training exited with code: {}".format(result.returncode))

    def forward(self, flags: argparse.Namespace):
        """
        Main function of the ``GridTrainerCPU``.

        Maps the grid experiments to CPU cores in the limit of the maximum concurrent runs allowed or maximum\
         available cores.

        :param flags: Parsed arguments from the parser.

        """
        # Ask for confirmation - optional.
        if flags.confirm:
            input('Press any key to continue')

        # Run in as many threads as there are CPUs available to the script
        max_processes = min(len(os.sched_getaffinity(0)), self.max_concurrent_run)

        with ThreadPool(processes=max_processes) as pool:
            func = partial(GridTrainerCPU.run_experiment, self, flags.episodic_trainer, self.outdir_str, prefix="")
            pool.map(func, self.experiments_list)

        self.logger.info('Grid training experiments finished.')


if __name__ == '__main__':
    # Create parser with list of  runtime arguments.
    argp = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # add default arguments
    worker.add_arguments(argp)

    # add grid trainers-specific arguments
    add_arguments(argp)

    # Parse arguments.
    FLAGS, unparsed = argp.parse_known_args()

    grid_trainer_cpu = GridTrainerCPU(FLAGS, cuda=False)
    grid_trainer_cpu.forward(FLAGS)
