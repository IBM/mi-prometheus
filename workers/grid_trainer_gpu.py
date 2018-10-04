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
grid_trainer_gpu.py:

    - This file contains the implementation of a worker spanning a grid of training experiments on\
     a collection of GPUs. It works by loading a template yaml file, modifying the resulting dict, and dumping\
      that as yaml into a temporary file. The ``Trainer`` is then executed using the temporary yaml file as the task.\
      It will run as many concurrent jobs as possible.

"""

__author__ = "Alexis Asseman, Younes Bouhadjar, Vincent Marois"

import os
import argparse
import subprocess
from time import sleep
from functools import partial
from multiprocessing.pool import ThreadPool

import workers.worker as worker
import workers.grid_trainer_cpu as gtc
from workers.grid_trainer_cpu import GridTrainerCPU


class GridTrainerGPU(GridTrainerCPU):
    """
    Grid Worker managing several training experiments on GPUs.

    Reuses the ``Trainer`` (can specify the base one or the episodic one) to start one experiment.

    Inherits from ``GridTrainerCPU`` as the constructor is identical.

    """
    def __init__(self, flags: argparse.Namespace, cuda=True):
        """
        Constructor for the ``GridTrainerGPU``:

            - Calls the constructor of ``GridTrainerCPU`` as it is identical.


        :param flags: Parsed arguments from the parser.

        :param cuda: Whether or not to use CUDA (cf ``GridTrainerGPU``). Default to True.
        :type cuda: bool

        """
        self.name = 'GridTrainerGPU'

        # call base constructor
        super(GridTrainerGPU, self).__init__(flags, cuda)

    def run_experiment(self, episodic_trainer, output_dir: str, experiment_configs: str):
        """
        Runs the specified experiment using one ``Trainer``.

        :param episodic_trainer: Whether to use the ``EpisodicTrainer`` instead of the default ``Trainer``
        :type episodic_trainer: bool

        :param output_dir: Output directory for the experiment files (logging, best model etc.)
        :type output_dir: str

        :param experiment_configs: Configuration file(s) passed to the trainer using its `--c` argument. If indicating\
        several config files, they must be separated with coma ",".
        :type experiment_configs: str

        ..note::

            Statistics exporting to TensorBoard is currently not activated.


        """
        # set the command to be executed using the indicated Trainer
        if episodic_trainer:
            command_str = "cuda-gpupick -n1 python3 workers/episodic_trainer.py --c {0} --outdir " + output_dir
        else:
            command_str = "cuda-gpupick -n1 python3 workers/trainer.py --c {0} --outdir " + output_dir

        # add experiment config
        command_str = command_str.format(experiment_configs)

        self.logger.info("Starting: {}".format(command_str))
        with open(os.devnull, 'w') as devnull:
            result = subprocess.run(command_str.split(" "), stdout=devnull)
        self.logger.info("Finished: {}".format(command_str))

        if result.returncode != 0:
            self.logger.info("Training exited with code: {}".format(result.returncode))

    def forward(self, flags: argparse.Namespace):
        """
        Main function of the ``GridTrainerGPU``.

        Maps the grid experiments to CUDA devices in the limit of the maximum concurrent runs allowed.

        :param flags: Parsed arguments from the parser.

        """
        # Ask for confirmation - optional.
        if flags.confirm:
            input('Press any key to continue')

        # Run in as many threads as there are GPUs available to the script
        with ThreadPool(processes=self.max_concurrent_run) as pool:
            # This contains a list of `AsyncResult` objects. To check if completed and get result.
            thread_results = []

            for task in self.experiments_list:
                func = partial(GridTrainerGPU.run_experiment, self, flags.episodic_trainer, self.outdir_str)
                thread_results.append(pool.apply_async(func, (task,)))

                # Check every 3 seconds if there is a (supposedly) free GPU to start a task on
                sleep(3)
                while [r.ready() for r in thread_results].count(False) >= self.max_concurrent_run:
                    sleep(3)

            # Equivalent of what would usually be called "join" for threads
            for r in thread_results:
                r.wait()


if __name__ == '__main__':
    # Create parser with list of  runtime arguments.
    argp = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # add default arguments
    worker.add_arguments(argp)

    # add grid trainers-specific arguments
    gtc.add_arguments(argp)

    # Parse arguments.
    FLAGS, unparsed = argp.parse_known_args()

    grid_trainer_gpu = GridTrainerGPU(FLAGS, cuda=True)
    grid_trainer_gpu.forward(FLAGS)
