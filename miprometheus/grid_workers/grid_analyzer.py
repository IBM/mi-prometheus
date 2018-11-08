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
import logging

import torch

import numpy as np
from datetime import datetime
from functools import partial
from multiprocessing.pool import ThreadPool

from miprometheus.grid_workers.grid_worker import GridWorker


class GridAnalyzer(GridWorker):
    """
    Implementation of the Grid Analyzer. Post-processes the test results of a grid of experiments and gather them in\
     a csv file.

    Inherits from ``GridWorker``.

    TODO: complete doc
    """

    def __init__(self, name="GridAnalyzer"):
        """
        Constructor for the ``GridAnalyzer``:

            - Calls basic constructor of ''GridWorker''

        :param name: Name of the worker (DEFAULT: "GridAnalyzer").
        :type name: str

        """
        # call base constructor
        super(GridAnalyzer, self).__init__(name=name,use_gpu=False)

    def check_if_file_exists(self, dir_, filename_):
        """ Function if file in directory exists

        :param dir_: Path to file.
        :param filename_: Name of the file to be opened and analysed.

        """
        return os.path.isfile(os.path.join(dir_, filename_))

    def check_file_content(self, dir_, filename_):
        """ Function checks if the number of lines in the file is > 1.

        :param dir_: Path to file.
        :param filename_: Name of the file to be opened and analysed.

        """
        return self.get_lines_number(os.path.join(dir_, filename_)) > 1

    def get_lines_number(self, filename_):
        """ Function gets number of lines in a given file.

        :param filename_: Name of the file (with path) to be opened and analysed.

        """
        with open(filename_) as f:
            return sum(1 for line in f)

    def get_experiment_tests(self, experiment_path_):
        """ Returns a list of folders with valid experiment tests.
        
        :param experiment_path_: Path to experiment (training) folder.

        """
        experiments_tests = []
        for root, dirs, _ in os.walk(experiment_path_, topdown=True):
            for name in dirs:
                experiments_tests.append(os.path.join(root, name))

        # Keep only the folders that contain `testing.csv`.
        experiments_tests = [elem for elem in experiments_tests if 
            self.check_if_file_exists(elem, 'testing_statistics.csv')]

        # Check if the files contain any collected training/validation statistics.
        experiments_tests = [elem for elem in experiments_tests if 
            self.check_file_content(elem, 'testing_statistics.csv')]
        return experiments_tests

    def setup_grid_experiment(self):
        """
        Setups the overall experiment:

        - Parses arguments and sets logger level.

        - Recursively creates the paths to the experiments folders, verifying that they contain \
        basic statistics files, i.e. `training_statistics.csv`, `validation_statistics.csv` and  \
        `testing_statistics.csv`.

        ..note::

            We also require that those files are valid, i.e. contain at least one line with \
            collected statistics (excluding the header).
        
        """
        # Parse arguments.
        self.flags, self.unparsed = self.parser.parse_known_args()

        # Set logger depending on the settings.
        self.logger.setLevel(getattr(logging, self.flags.log_level.upper(), None))

        # Check if experiments directory was indicated.
        if self.flags.outdir == '':
            print('Please pass the experiments directory as --outdir')
            exit(-1)

        # Get experiment directory.        
        self.experiment_rootdir = self.flags.outdir

        # Get all sub-directories paths in outdir.
        self.experiments_list = []

        for root, dirs, _ in os.walk(self.experiment_rootdir, topdown=True):
            for name in dirs:
                self.experiments_list.append(os.path.join(root, name))

        # Keep only the folders that contain training_configuration.yaml, training_statistics.csv and
        # training.csv and model (which contains aggregated validation statistics).
        self.experiments_list = [elem for elem in self.experiments_list if 
            self.check_if_file_exists(elem, 'training_configuration.yaml') and
            self.check_if_file_exists(elem, 'training_statistics.csv') and
            self.check_if_file_exists(elem, 'models/model_best.pt')]

        # Check if the training statistics file contain any records.
        self.experiments_list = [elem for elem in self.experiments_list if 
             self.check_file_content(elem, 'training_statistics.csv')]

        # Check if there are some valid folders.
        if len(self.experiments_list) == 0:
            self.logger.error("There are no valid experiments in {} directory!".format(self.experiment_rootdir))
            exit(-2)

        # List folders with "valid" experiments.
        exp_str = "Found the following valid experiments in {} directory:\n".format(self.experiment_rootdir)
        exp_str += '='*80 + '\n'
        for exp in self.experiments_list:
            exp_str += " - {}\n".format(exp)
        exp_str += '='*80 + '\n'
        self.logger.info(exp_str)

        # Detect how many tests runs have been done for each experiments/models.
        number_of_test = []
        for exp_path in self.experiments_list:
            # Get all tests for a given training experiment.
            experiments_tests = self.get_experiment_tests(exp_path)

            # Save number of tests.
            number_of_test.append(len(experiments_tests))

        # Not sure why we need that :]
        # Assert that the number is the same for all experiments.
        if len(set(number_of_test)) != 1:
            self.logger.error('Not all experiments have the same number of tests statistics collected!')
            exit(-3)
        
        # Get number of tests.
        self.num_tests = number_of_test[0]

        # Check number of tests.
        if self.num_tests < 1:
            self.logger.error('Tests statistics not found! (hint: please use mip-tester/mip-grid-tester to collect them)')
            exit(-4)
        # Ok, proceed.
        self.logger.info('Detected number of tests per experiment: {}'.format(self.num_tests))


    def run_experiment(self, experiment_path: str):
        """
        Analyzes test results.

        TODO: complete doc

        :param experiment_path: Path to an experiment folder containing a trained model.
        :type experiment_path: str

        """
        # Create results dictionary.
        r = dict()  


        # Load yaml file. To get model name, problem name and random seeds
        with open(os.path.join(experiment_path, 'training_configuration.yaml'), 'r') as yaml_file:
            params = yaml.load(yaml_file)

        # Get problem and model names - from config.
        r['problem'] = params['testing']['problem']['name']
        r['model'] = params['model']['name']

        # Load checkpoint from model file.
        chkpt = torch.load(os.path.join(experiment_path, 'models/model_best.pt'),
            map_location=lambda storage, loc: storage)

        # Get episode - from checkpoint.
        episode = int(chkpt['stats']['episode'])

        # Copy training status - from checkpoint.
        r['train_status'] = chkpt['status']

        r['train_start_timestamp'] = os.path.basename(os.path.normpath(experiment_path))
        #r['train_seed_torch'] = params['training']['seed_torch']
        #r['train_seed_numpy'] = params['training']['seed_numpy']

        with open(os.path.join(experiment_path, 'training_statistics.csv'), mode='r') as f:
            train_reader = csv.DictReader(f)
            # Get row with episode.
            for row in train_reader: 
                # Iterate over column name and value.
                if int(row['episode']) == episode:
                    # Copy training statistics.
                    for key, value in row.items():
                        r['train_{}'.format(key)] = str(value)          
                    


        r['valid_timestamp'] = '{0:%Y%m%d_%H%M%S}'.format(chkpt['timestamp']) # str(chkpt['timestamp'])
        # Copy other values and add 'valid_' prefices from 
        for key, value in chkpt['stats'].items():
            r['valid_{}'.format(key)] = str(value)          

        # Ok, return what we got right now.
        return r

        # Get all tests for a given training experiment.
        experiments_tests = self.get_experiment_tests(experiment_path)

        with open(os.path.join(experiment_path, 'validation_statistics.csv'), mode='r') as f:
            valid_csv = csv.reader(f, delimiter=',')


        #for row in train_csv:
            #print(', '.join(row))
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
        for test_idx, experiment in zip(range(1, self.num_tests+1), experiments_tests):

            with open(os.path.join(experiment_path, 'testing_statistics.csv'), mode='r') as f:
                test_csv = csv.reader(f, delimiter=',')

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
        # Go throught experiments one by one and collect data.
        list_dict_exp = []
        for exp in self.experiments_list:
            print(exp)
            list_dict_exp.append(self.run_experiment(exp))

        print(list_dict_exp)
        exp_values = dict(zip(list_dict_exp[0], zip(*[d.values() for d in list_dict_exp])))

        # create results file
        results_file = os.path.join(self.experiment_rootdir, "{0:%Y%m%d_%H%M%S}_grid_analysis.csv".format(datetime.now()))

        with open(results_file, "w") as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerow(exp_values.keys())
            writer.writerows(zip(*exp_values.values()))

        self.logger.info('Analysis done.')
        self.logger.info('Results stored in {}.'.format(results_file))


def main():
    """
    Entry point function for the ``GridAnalyzer``.

    """
    grid_analyzer = GridAnalyzer()

    # parse args, load configuration and create all required objects.
    grid_analyzer.setup_grid_experiment()

    # GO!
    grid_analyzer.run_grid_experiment()


if __name__ == '__main__':

    main()
