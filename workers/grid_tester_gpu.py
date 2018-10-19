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
grid_tester_gpu.py:

    - This file contains the implementation of a worker running the ``Tester`` on the results of a ``GridTrainer``\
     using CPUs.

    - The input is a list of directories for each problem/model e.g. `experiments/serial_recall/dnc`, \
      and executes on every run of the model in that directory.

"""

__author__ = "Tomasz Kornuta & Vincent Marois"

import torch
from time import sleep
from functools import partial
from multiprocessing.pool import ThreadPool

from workers.grid_tester_cpu import GridTesterCPU


class GridTesterGPU(GridTesterCPU):
    """
    Implementation of the ``GridTester`` running on GPUs.

    Reuses the ``Tester`` to start one test experiment.

    Inherits from ``GridTesterCPU`` as the constructor is identical.

    """

    def __init__(self, name="GridTesterGPU", use_gpu=True):
        """
        Constructor for the ``GridTesterGPU``:

            - Calls the constructor of ``GridTesterCPU`` as it is identical.


        :param name: Name of the worker (DEFAULT: "GridTesterGPU").
        :type name: str

        :param use_gpu: Indicates whether the worker should use GPU or not.
        :type use_gpu: bool

        """
        # call base constructor
        super(GridTesterGPU, self).__init__(name=name,use_gpu=use_gpu)

    def run_grid_experiment(self):
        """
        Main function of the ``GridTesterGPU``.

        Maps the grid experiments to CPU cores in the limit of the maximum concurrent runs allowed or maximum\
         available cores.

        """
        # Ask for confirmation - optional.
        if self.flags.confirm:
            input('Press any key to continue')

        # Run in as many threads as there are GPUs available to the script
        with ThreadPool(processes=torch.cuda.device_count()) as pool:
            # This contains a list of `AsyncResult` objects. To check if completed and get result.
            thread_results = []

            for task in self.experiments_list:
                func = partial(GridTesterGPU.run_experiment, self, prefix="cuda-gpupick -n1 ")
                thread_results.append(pool.apply_async(func, (task,)))

                # Check every 3 seconds if there is a (supposedly) free GPU to start a task on
                sleep(3)
                while [r.ready() for r in thread_results].count(False) >= torch.cuda.device_count():
                    sleep(3)

            # Equivalent of what would usually be called "join" for threads
            for r in thread_results:
                r.wait()

        self.logger.info('Grid training experiments finished.')


if __name__ == '__main__':
    grid_tester_gpu = GridTesterGPU()

    # parse args, load configuration and create all required objects.
    grid_tester_gpu.setup_grid_experiment()

    # GO!
    grid_tester_gpu.run_grid_experiment()
