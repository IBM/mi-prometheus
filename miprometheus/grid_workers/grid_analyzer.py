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
import torch
import logging
from datetime import datetime

from miprometheus.grid_workers.grid_worker import GridWorker


class GridAnalyzer(GridWorker):
    """
    Implementation of the :py:class:`miprometheus.grid_workers.GridAnalyzer`.

    Post-processes the test results of a grid of experiments and gather them in a csv file.

    This csv file will gather the training statistics (seeds, accuracies, terminal conditions,...), \
    the validation statistics and the test statistics.

    Inherits from :py:class:`miprometheus.grid_workers.GridWorker`.

    """

    def __init__(self, name="GridAnalyzer"):
        """
        Constructor for the :py:class:`miprometheus.grid_workers.GridAnalyzer`:

            - Calls basic constructor of :py:class:`miprometheus.grid_workers.GridWorker`

        :param name: Name of the worker (DEFAULT: "GridAnalyzer").
        :type name: str

        """
        # call base constructor
        super(GridAnalyzer, self).__init__(name=name, use_gpu=False)

    @staticmethod
    def check_if_file_exists(dir_, filename_):
        """
        Checks if ``filename_`` exists in ``dir_``.

        :param dir_: Path to file.
        :type dir_: str

        :param filename_: Name of the file to be opened and analysed.
        :type filename_: str

        :return: True if the file exists in the directory, else False

        """
        return os.path.isfile(os.path.join(dir_, filename_))

    def check_file_content(self, dir_, filename_):
        """
        Checks if the number of lines in the file is > 1.

        :param dir_: Path to file.
        :type dir_: str

        :param filename_: Name of the file to be opened and analyzed.
        :type filename_: str

        :return: True if the number of lines in the file is strictly greater than one.

        """
        return self.get_lines_number(os.path.join(dir_, filename_)) > 1

    @staticmethod
    def get_lines_number(filename_):
        """
        Returns the number of lines in ``filename_``.

        :param filename_: Filepath to be opened and line-read.
        :type filename_: str

        :return: Number of lines in the file.

        """
        with open(filename_) as f:
            return sum(1 for _ in f)

    def get_experiment_tests(self, experiment_path_):
        """
        Returns a list of folders containing valid test experiments data:

            - A configuration (`testing_configuration.yaml`),
            - A csv file containing a data point for the aggregated statistics (`testing_set_agg_statistics.csv`)

        
        :param experiment_path_: Path to experiment (training) folder.
        :type experiment_path_: str

        :return: A list of valid test experiment folders.

        """
        experiments_tests = []
        for root, dirs, _ in os.walk(experiment_path_, topdown=True):
            for name in dirs:
                experiments_tests.append(os.path.join(root, name))

        # Keep only the folders that contain a test configuration file and a csv statistics file.
        experiments_tests = [elem for elem in experiments_tests if 
                             self.check_if_file_exists(elem, 'testing_configuration.yaml') and
                             self.check_if_file_exists(elem, 'testing_set_agg_statistics.csv')]

        # Check if the csv file contains at least one data point.
        experiments_tests = [elem for elem in experiments_tests if 
                             self.check_file_content(elem, 'testing_set_agg_statistics.csv')]

        return experiments_tests

    def setup_grid_experiment(self):
        """
        Setups the overall experiment:

            - Parses arguments and sets logger level,
            - Checks the presence of experiments folder,
            - Recursively traverses the experiment folders, cherry-picking subfolders containing:

                - (a) 'training_configuration.yaml' (training configuration file),
                - (b) 'models/model_best.pt' (checkpoint of the best saved model).


        """
        # Parse arguments.
        self.flags, self.unparsed = self.parser.parse_known_args()

        # Set logger depending on the settings.
        self.logger.setLevel(getattr(logging, self.flags.log_level.upper(), None))

        # Check if experiments directory was indicated.
        if self.flags.expdir == '':
            print('Please pass the experiments directory as --expdir')
            exit(-1)

        # Get experiment directory.
        self.experiment_rootdir = self.flags.expdir

        # Get all sub-directories paths in expdir.
        self.experiments_list = []

        for root, dirs, _ in os.walk(self.experiment_rootdir, topdown=True):
            for name in dirs:
                self.experiments_list.append(os.path.join(root, name))

        # Keep only the folders that contain training_configuration.yaml, training_statistics.csv and
        # training.csv and model (which contains aggregated validation statistics).
        self.experiments_list = [elem for elem in self.experiments_list if 
                                 self.check_if_file_exists(elem, 'training_configuration.yaml') and
                                 self.check_if_file_exists(elem, 'models/model_best.pt')]

        # Check if there are some valid folders.
        if len(self.experiments_list) == 0:
            self.logger.error("There are no valid experiment folders in {} directory!".format(self.experiment_rootdir))
            exit(-2)

        # List folders with "valid" experiment data.
        exp_str = "Found the following valid experiments in directory: {} \n".format(self.experiment_rootdir)
        exp_str += '='*80 + '\n'
        for exp in self.experiments_list:
            exp_str += " - {}\n".format(exp)
        exp_str += '='*80 + '\n'
        self.logger.info(exp_str)

        # Ask for confirmation - optional.
        if self.flags.user_confirm:
            try:
                input('Press <Enter> to confirm and start the grid analyzis\n')
            except KeyboardInterrupt:
                exit(0)    

    def run_experiment(self, experiment_path: str):
        """
        Collects the training / validation / test statistics for a given experiment path.

        Analyzes whether the given training experiment folder contains subfolders with test experiments data:

            - Loads and parses training configuration file,
            - Loads checkpoint with model and training and validation statistics,
            - Recursively traverses subdirectories looking for test experiments,

              .. note::

                We require that the test statistics csv files are valid, i.e. contain at least one line with \
                collected statistics (excluding the header).


            - Collects statistics from training, validation (from model checkpoint) and test experiments \
              (from test csv files found in subdirectories).

        :param experiment_path: Path to an experiment folder containing a training statistics.
        :type experiment_path: str

        :return: Four dictionaries containing:
                    - Status info (model, problem etc.),
                    - Training statistics,
                    - Validation statistics,
                    - Test statistics.


        """
        self.logger.info('Analyzing experiments from: {}'.format(experiment_path))

        # Create dictionaries.
        status_dict = dict()
        train_dict = dict()
        valid_dict = dict()

        # Load yaml file, to get model name, problem name and random seeds.
        with open(os.path.join(experiment_path, 'training_configuration.yaml'), 'r') as yaml_file:
            params = yaml.load(yaml_file)

        # Get problem and model names - from config.
        status_dict['problem'] = params['testing']['problem']['name']
        status_dict['model'] = params['model']['name']

        # Load checkpoint from model file.
        chkpt = torch.load(os.path.join(experiment_path, 'models/model_best.pt'),
                           map_location=lambda storage, loc: storage)

        status_dict['model_save_timestamp'] = '{0:%Y%m%d_%H%M%S}'.format(chkpt['model_timestamp']) 
        status_dict['training_terminal_status'] = chkpt['status']
        status_dict['training_terminal_status_timestamp'] = '{0:%Y%m%d_%H%M%S}'.format(chkpt['status_timestamp'])


        # Create "empty" equivalent.
        status_dict_empty = dict.fromkeys(status_dict.keys(), ' ')

        # Copy training status stats.
        train_dict['training_configuration_filepath'] = os.path.join(experiment_path, 'training_configuration.yaml')
        train_dict['training_start_timestamp'] = os.path.basename(os.path.normpath(experiment_path))
        train_dict['training_seed_torch'] = params['training']['seed_torch']
        train_dict['training_seed_numpy'] = params['training']['seed_numpy']                    

        # Copy the training statistics from the checkpoint and add the 'train_' prefix.
        for key, value in chkpt['training_stats'].items():
            train_dict['training_{}'.format(key)] = value
        # Create "empty" equivalent.
        train_dict_empty = dict.fromkeys(train_dict.keys(), ' ')

        # Copy the validation statistics from the checkpoint and add the 'valid_' prefix.
        for key, value in chkpt['validation_stats'].items():
            valid_dict['validation_{}'.format(key)] = value
        # Create "empty" equivalent.
        valid_dict_empty = dict.fromkeys(valid_dict.keys(), ' ')

        # Get all tests for a given training experiment.
        experiments_tests = self.get_experiment_tests(experiment_path)

        list_test_dicts = []

        if len(experiments_tests) > 0:
            self.logger.info('  - Found {} test(s)'.format(len(experiments_tests)))

            # "Expand" status, train and valid dicts by empty ones, prop. to the number of test folders.
            list_status_dicts = [status_dict, *[status_dict_empty for _ in range(len(experiments_tests) - 1)]]
            list_train_dicts = [train_dict, *[train_dict_empty for _ in range(len(experiments_tests) - 1)]]
            list_valid_dicts = [valid_dict, *[valid_dict_empty for _ in range(len(experiments_tests) - 1)]]

            # Get tests statistics.
            for experiment_test_path in experiments_tests:
                self.logger.info('  - Analyzing test from: {}'.format(experiment_test_path))

                # Create test dict:
                test_dict = dict()
                test_dict['test_configuration_filepath'] = os.path.join(experiment_test_path, 'testing_set_agg_statistics.yaml')
                test_dict['test_start_timestamp'] = os.path.basename(os.path.normpath(experiment_test_path))[5:]

                # Load yaml file and get random seeds.
                with open(os.path.join(experiment_test_path, 'testing_configuration.yaml'), 'r') as yaml_file:
                    test_params = yaml.load(yaml_file)  
                    # Get seeds.             
                    test_dict['test_seed_torch'] = test_params['testing']['seed_torch']
                    test_dict['test_seed_numpy'] = test_params['testing']['seed_numpy']                    

                # Load csv file and copy test statistics
                with open(os.path.join(experiment_test_path, 'testing_set_agg_statistics.csv'), mode='r') as f:
                    # Open file.
                    test_reader = csv.DictReader(f)

                    # Copy training statistics.
                    for row in test_reader:
                        for key, value in row.items():
                            test_dict['test_{}'.format(key)] = value   

                list_test_dicts.append(test_dict)

        else:
            self.logger.info('  - Could not find any valid tests')
            list_status_dicts = [status_dict]
            list_train_dicts = [train_dict]
            list_valid_dicts = [valid_dict]

            # Add "empty test entry"
            list_test_dicts.append({})

        # Return all dictionaries with lists
        return list_status_dicts, list_train_dicts, list_valid_dicts, list_test_dicts

    @staticmethod
    def merge_list_dicts(list_dicts):
        """
        Merges a list of dictionaries by filling the missing fields with spaces into one dict.
        
        :param list_dicts: List of dictionaries, potentially containing different headers, which will be merged.
        :type list_dicts: list

        :return: dict, resulting of the merge.

        """
        # Create a "unified" header.
        header = set(k for d in list_dicts for k in d)

        # Create an "empty" dict from the unified header.
        empty_dict = {k: ' ' for k in header}

        # "Fill" all lists with empty gaps.
        list_filled_dicts = []
        for i, _ in enumerate(list_dicts):
            list_filled_dicts.append({**empty_dict, **(list_dicts[i])})

        # Zip lists of dicts.
        final_dict = dict(zip(header, zip(*[d.values() for d in list_filled_dicts])))

        # Return the result.
        return final_dict

    def run_grid_experiment(self):
        """
        Collects four list of dicts from each experiment path contained in ``self.experiments_lists``.

        Merges all them together and saves result to a single csv file.

        """
        try:
            # Go through the experiments one by one and collect data.
            list_statuses = []
            list_trains = []
            list_valids = []
            list_tests = []

            for exp in self.experiments_list:
                statuses, trains, valids, tests = self.run_experiment(exp)
                list_statuses.extend(statuses)
                list_trains.extend(trains)
                list_valids.extend(valids)
                list_tests.extend(tests)

            # Merge lists.
            statuses = self.merge_list_dicts(list_statuses)
            trains = self.merge_list_dicts(list_trains)
            valids = self.merge_list_dicts(list_valids)
            tests = self.merge_list_dicts(list_tests)

            # Merge everything into one big dictionary..
            exp_values = {**statuses, **trains, **valids, **tests}

            # create results file
            results_file = os.path.join(self.experiment_rootdir,
                                        "{0:%Y%m%d_%H%M%S}_grid_analysis.csv".format(datetime.now()))

            with open(results_file, "w") as outfile:
                writer = csv.writer(outfile, delimiter=',')
                writer.writerow(exp_values.keys())
                writer.writerows(zip(*exp_values.values()))

            self.logger.info('Analysis finished')
            self.logger.info('Results stored in {}.'.format(results_file))

        except KeyboardInterrupt:
            self.logger.info('Grid analysis interrupted!')


def main():
    """
    Entry point function for the :py:class:`miprometheus.grid_workers.GridAnalyzer`.

    """
    grid_analyzer = GridAnalyzer()

    # parse args, load configuration and create all required objects.
    grid_analyzer.setup_grid_experiment()

    # GO!
    grid_analyzer.run_grid_experiment()


if __name__ == '__main__':

    main()
