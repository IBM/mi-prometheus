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

    - This file contains the implementation of a worker running the ``Tester`` on the results of a ``GridTrainer``
    using CPUs.

    - The input is a list of directories for each problem/model e.g. `experiments/serial_recall/dnc`, \
      and executes on every run of the model in that directory.

"""

__author__ = "Tomasz Kornuta & Vincent Marois"

import argparse
from time import sleep
from functools import partial
from multiprocessing.pool import ThreadPool

import workers.worker as worker
import workers.grid_tester_cpu as gtc
from workers.grid_tester_cpu import GridTesterCPU


MAX_THREADS = 6


class GridTesterGPU(GridTesterCPU):
    """
    Implementation of the Grid Tester running on GPUs.

    Reuses the ``Tester`` to start one test experiment.

    Inherits from ``GridTesterCPU`` as the constructor is identical.

    """

    def __init__(self, flags: argparse.Namespace):
        """
        Constructor for the ``GridTesterGPU``:

            - Calls the constructor of ``GridTrainerCPU`` as it is identical.


        :param flags: Parsed arguments from the parser.

        """
        self.name = 'GridTesterGPU'

        # call base constructor
        super(GridTesterCPU, self).__init__(flags)

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

        # Run in as many threads as there are GPUs available to the script
        with ThreadPool(processes=MAX_THREADS) as pool:
            # This contains a list of `AsyncResult` objects. To check if completed and get result.
            thread_results = []

            for task in self.experiments_list:
                func = partial(GridTesterGPU.run_experiment, self, prefix="cuda-gpupick -n1 ")
                thread_results.append(pool.apply_async(func, (task,)))

                # Check every 3 seconds if there is a (supposedly) free GPU to start a task on
                sleep(3)
                while [r.ready() for r in thread_results].count(False) >= MAX_THREADS:
                    sleep(3)

            # Equivalent of what would usually be called "join" for threads
            for r in thread_results:
                r.wait()

        self.logger.info('Grid training experiments finished.')


if __name__ == '__main__':
    # Create parser with list of  runtime arguments.
    argp = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # add default arguments
    worker.add_arguments(argp)

    # add grid trainers-specific arguments
    gtc.add_arguments(argp)

    # Parse arguments.
    FLAGS, unparsed = argp.parse_known_args()

    grid_tester_gpu = GridTesterGPU(FLAGS)
    grid_tester_gpu.forward(FLAGS)
