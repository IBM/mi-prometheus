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

import shutil
import torch
from time import sleep
from functools import partial
from multiprocessing.pool import ThreadPool

from miprometheus.grid_workers.grid_trainer_cpu import GridTrainerCPU


class GridTrainerGPU(GridTrainerCPU):
    """
    Grid Worker managing several training experiments on GPUs.

    Reuses a ``Trainer`` (can specify the ``classic`` one or the ``flexible`` one) to start one experiment.

    Inherits from ``GridTrainerCPU`` as the constructor & ``setup_grid_experiment`` are identical.

    """
    def __init__(self, name="GridTrainerGPU", use_gpu=True):
        """
        Constructor for the ``GridTrainerGPU``:

            - Calls the constructor of ``GridTrainerCPU`` as it is identical.


        :param name: Name of the worker (DEFAULT: "GridTrainerGPU").
        :type name: str

        :param use_gpu: Indicates whether the worker should use GPU or not.
        :type use_gpu: bool

        """
        # Call the base constructor.
        super(GridTrainerGPU, self).__init__(name=name,use_gpu=use_gpu)


    def setup_grid_experiment(self):
        """
        Setups a specific experiment.

        - Calls the ``super(self).setup_experiment()`` to parse arguments, parse config files etc.

        - Checks the presence of CUDA-compatible devices.

        """
        super(GridTrainerGPU, self).setup_grid_experiment()
        # Check the presence of the CUDA-compatible devices.
        if (torch.cuda.device_count() == 0):
            self.logger.error("Cannot use GPU as there are no CUDA-compatible devices present in the system!")
            exit(-1)


    def run_grid_experiment(self):
        """
        Main function of the ``GridTrainerGPU``.

        Maps the grid experiments to CUDA devices in the limit of the maximum concurrent runs allowed.

        """
        try:

            # Check the presence of cuda-gpupick
            if shutil.which('cuda-gpupick') is not None:
                prefix_str = "cuda-gpupick -n1 "
            else:
                self.logger.warning("Cannot localize the 'cuda-gpupick' script, disabling it")
                prefix_str = ''

            # Check max number of child processes. 
            if self.max_concurrent_runs <= 0: # We need at least one proces!
                max_processes = torch.cuda.device_count()
            else:    
                # Take into account the minimum value.
                max_processes = min(torch.cuda.device_count(), self.max_concurrent_runs)
            self.logger.info('Spanning experiments using {} GPU(s) concurrently.'.format(max_processes))

            # Run in as many threads as there are GPUs available to the script.
            with ThreadPool(processes=max_processes) as pool:
                # This contains a list of `AsyncResult` objects. To check if completed and get result.
                thread_results = []

                for task in self.experiments_list:
                    func = partial(GridTrainerGPU.run_experiment, self, prefix=prefix_str)
                    thread_results.append(pool.apply_async(func, (task,)))

                    # Check every 3 seconds if there is a (supposedly) free GPU to start a task on
                    sleep(3)
                    while [r.ready() for r in thread_results].count(False) >= max_processes:
                        sleep(3)

                # Equivalent of what would usually be called "join" for threads
                for r in thread_results:
                    r.wait()

            self.logger.info('Grid training finished')

        except KeyboardInterrupt:
            self.logger.info('Grid training interrupted!')


def main():
    """
    Entry point function for the ``GridTrainerGPU``.

    """
    grid_trainer_gpu = GridTrainerGPU()

    # parse args, load configuration and create all required objects.
    grid_trainer_gpu.setup_grid_experiment()

    # GO!
    grid_trainer_gpu.run_grid_experiment()


if __name__ == '__main__':

    main()
