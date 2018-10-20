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
grid_analyzer.py:

    - This script post-processes the output of the ``GridTrainers`` and ``GridTesters``. \
    It gathers the test results into one `.csv` file.


"""
__author__ = "Tomasz Kornuta & Vincent Marois"

import os
import csv
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from functools import partial
from multiprocessing.pool import ThreadPool

from mip.workers.grid_worker import GridWorker


class GridAnalyzer(GridWorker):
    """
    Implementation of the Grid Analyzer. Post-processes the test results of a grid of experiments and gather them in\
     a csv file.

    Inherits from ``GridWorker``.

    TODO: complete doc
    """

    def __init__(self, name="GridAnalyzer", use_gpu=False):
        """
        Constructor for the ``GridAnalyzer``:

            - TODO: complete doc


        :param name: Name of the worker (DEFAULT: "GridAnalyzer").
        :type name: str

        :param use_gpu: Indicates whether the worker should use GPU or not.
        :type use_gpu: bool

        """
        # call base constructor
        super(GridAnalyzer, self).__init__(name=name,use_gpu=use_gpu)

    def setup_grid_experiment(self):
        """
        Setups the overall experiment:

        - Calls the ``super(self).setup_experiment()`` to parse arguments,

        - Recursively creates the paths to the experiments folders, verifying that they are valid (e.g. \
        contain `training_statistics.csv`, `validation_statistics.csv` and `testing_statistics.csv`).

        """
        super(GridAnalyzer, self).setup_grid_experiment()

        # Check if experiments directory was indicated.
        if self.flags.outdir == '':
            print('Please pass the experiments directory as --outdir')
            exit(-1)

        self.directory_chckpnts = self.flags.outdir

        # get all sub-directories paths in outdir
        self.experiments_list = []

        for root, dirs, files in os.walk(self.directory_chckpnts, topdown=True):
            for name in dirs:
                self.experiments_list.append(os.path.join(root, name))

        # Keep only the folders that contain validation.csv and training.csv
        self.experiments_list = [elem for elem in self.experiments_list if os.path.isfile(
            elem + '/validation_statistics.csv') and os.path.isfile(elem + '/training_statistics.csv')]

        # check if the files are empty except for the first line
        self.experiments_list = [elem for elem in self.experiments_list if os.stat(
            elem + '/validation_statistics.csv').st_size > 24 and os.stat(elem + '/training_statistics.csv').st_size > 24]

        # the following is to detect how many tests runs have been done for each experiment,
        # and asserting that the number is the same for all experiment
        number_of_test = []
        for path in self.experiments_list:
            experiments_tests = []
            for root, dirs, files in os.walk(path, topdown=True):
                for name in dirs:
                    experiments_tests.append(os.path.join(root, name))

            # Keep only the folders that contain `testing.csv`
            experiments_tests = [elem for elem in experiments_tests if os.path.isfile(elem + '/testing_statistics.csv')]

            # check if that the `testing.csv` files are not empty
            experiments_tests = [elem for elem in experiments_tests if os.stat(elem + '/testing_statistics.csv').st_size > 24]
            number_of_test.append(len(experiments_tests))

        assert len(set(number_of_test)) == 1, 'Not all experiments have the same number of tests'
        self.nb_tests = number_of_test[0]
        self.logger.info('Detected a number of tests per experiment of {}.'.format(self.nb_tests))

    def run_experiment(self, experiment_path: str):
        """
        Analyzes test results.

        TODO: complete doc

        :param experiment_path: Path to an experiment folder containing a trained model.
        :type experiment_path: str


        ..note::

            Visualization is deactivated to avoid any user interaction.

            TODO: anything else?


        """
        r = dict()  # results dictionary

        r['timestamp'] = os.path.basename(os.path.normpath(experiment_path))

        # Load yaml file. To get model name, problem name and random seeds
        with open(experiment_path + '/training_configuration.yaml', 'r') as yaml_file:
            params = yaml.load(yaml_file)

        r['model'] = params['model']['name']
        r['problem'] = params['testing']['problem']['name']

        r['seed_torch'] = params['training']['seed_torch']
        r['seed_numpy'] = params['training']['seed_numpy']

        # get all sub-directories paths in experiment_path: to detect test experiments paths
        experiments_tests = []

        for root, dirs, files in os.walk(experiment_path, topdown=True):
            for name in dirs:
                experiments_tests.append(os.path.join(root, name))

        # Keep only the folders that contain `testing.csv`
        experiments_tests = [elem for elem in experiments_tests if os.path.isfile(elem + '/testing_statistics.csv')]

        # check if that the `testing.csv` files are not empty
        experiments_tests = [elem for elem in experiments_tests if os.stat(elem + '/testing_statistics.csv').st_size > 24]

        valid_csv = pd.read_csv(experiment_path + '/validation_statistics.csv', delimiter=',', header=0)
        train_csv = pd.read_csv(experiment_path + '/training_statistics.csv', delimiter=',', header=0)

        # get best train point
        train_episode = train_csv.episode.values.astype(int)
        train_loss = train_csv.loss.values.astype(float)

        index_train_loss = np.argmin(train_loss)
        r['best_train_ep'] = train_episode[index_train_loss]  # episode index of lowest training loss
        r['training_episodes_limit'] = train_episode[-1]
        r['best_train_loss'] = train_loss[index_train_loss]  # lowest training loss

        if 'acc' in train_csv:
            train_accuracy = train_csv.acc.values.astype(float)
            r['best_train_acc'] = train_accuracy[index_train_loss]

        # best valid point
        valid_episode = valid_csv.episode.values.astype(int)
        valid_loss = valid_csv.loss.values.astype(float)

        index_val_loss = np.argmin(valid_loss)
        r['best_valid_ep'] = valid_episode[index_val_loss]  # episode index of lowest validation loss
        r['best_valid_loss'] = valid_loss[index_val_loss]  # lowest validation loss

        if 'acc' in valid_csv:
            valid_accuracy = valid_csv.acc.values.astype(float)
            r['best_valid_accuracy'] = valid_accuracy[index_val_loss]

        # get test statistics
        for test_idx, experiment in zip(range(1, self.nb_tests+1), experiments_tests):

            test_csv = pd.read_csv(experiment + '/testing_statistics.csv', delimiter=',', header=0)
            # get average test loss
            nb_episode = test_csv.episode.values.astype(int)[-1]+1
            losses = test_csv.loss.values.astype(float)

            r['test_{}_average_loss'.format(test_idx)] = sum(losses)/nb_episode
            r['test_{}_std_loss'.format(test_idx)] = np.std(losses)

            if 'acc' in test_csv:
                accuracies = test_csv.acc.values.astype(float)
                r['test_{}_average_acc'.format(test_idx)] = sum(accuracies) / nb_episode
                r['test_{}_std_acc'.format(test_idx)] = np.std(accuracies)

        return r

    def run_grid_experiment(self):
        """
        Constructor for the ``GridAnalyzer``.

        Maps the grid analysis to CPU cores in the limit of the available cores.


        """
        # Run in as many threads as there are CPUs available to the script
        with ThreadPool(processes=len(os.sched_getaffinity(0))) as pool:
            func = partial(GridAnalyzer.run_experiment, self)
            list_dict_exp = pool.map(func, self.experiments_list)

            exp_values = dict(zip(list_dict_exp[0], zip(*[d.values() for d in list_dict_exp])))

            # create results file
            results_file = os.path.join(self.directory_chckpnts, "{0:%Y%m%d_%H%M%S}_grid_analysis.csv".format(datetime.now()))

            with open(results_file, "w") as outfile:
                writer = csv.writer(outfile, delimiter=',')
                writer.writerow(exp_values.keys())
                writer.writerows(zip(*exp_values.values()))

        self.logger.info('Analysis done.')
        self.logger.info('Results stored in {}.'.format(results_file))


def main():
    grid_analyzer = GridAnalyzer()

    # parse args, load configuration and create all required objects.
    grid_analyzer.setup_grid_experiment()

    # GO!
    grid_analyzer.run_grid_experiment()


if __name__ == '__main__':

    main()
