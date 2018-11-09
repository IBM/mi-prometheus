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
            self.check_if_file_exists(elem, 'testing_configuration.yaml') and
            self.check_if_file_exists(elem, 'testing_set_agg_statistics.csv') ]

        # Check if the files contain any collected training/validation statistics.
        experiments_tests = [elem for elem in experiments_tests if 
            self.check_file_content(elem, 'testing_set_agg_statistics.csv')]
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
        #    self.check_if_file_exists(elem, 'training_statistics.csv') and
            self.check_if_file_exists(elem, 'models/model_best.pt')]

        # Check if the training statistics file contain any records.
        #self.experiments_list = [elem for elem in self.experiments_list if 
        #     self.check_file_content(elem, 'training_statistics.csv')]

        # Check if there are some valid folders.
        if len(self.experiments_list) == 0:
            self.logger.error("There are no valid experiments in {} directory!".format(self.experiment_rootdir))
            exit(-2)

        # List folders with "valid" experiments.
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
            except Exception:
                pass            
            except KeyboardInterrupt:
                exit(0)    

    def run_experiment(self, experiment_path: str):
        """
        Analyzes test results.

        TODO: complete doc

        :param experiment_path: Path to an experiment folder containing a trained model.
        :type experiment_path: str

        """
        self.logger.info('Analysing experiment from: {}'.format(experiment_path))

        # Create dictionaries.
        status_dict = dict()
        train_dict = dict()
        valid_dict = dict()

        # Load yaml file. To get model name, problem name and random seeds
        with open(os.path.join(experiment_path, 'training_configuration.yaml'), 'r') as yaml_file:
            params = yaml.load(yaml_file)

        # Get problem and model names - from config.
        status_dict['problem'] = params['testing']['problem']['name']
        status_dict['model'] = params['model']['name']

        # Load checkpoint from model file.
        chkpt = torch.load(os.path.join(experiment_path, 'models/model_best.pt'),
            map_location=lambda storage, loc: storage)

        status_dict['model_timestamp'] = '{0:%Y%m%d_%H%M%S}'.format(chkpt['timestamp']) 
        status_dict['train_status'] = chkpt['status']

        # Create "empty" equivalent.
        status_dict_empty = dict.fromkeys(status_dict.keys(), ' ')

        # Copy training status stats.
        train_dict['train_config'] = os.path.join(experiment_path, 'training_configuration.yaml')
        train_dict['train_start'] = os.path.basename(os.path.normpath(experiment_path))
        train_dict['train_seed_torch'] = params['training']['seed_torch']
        train_dict['train_seed_numpy'] = params['training']['seed_numpy']                    

        # Copy training statistics and add 'valid_' prefices from 
        for key, value in chkpt['training_stats'].items():
            train_dict['train_{}'.format(key)] = value
        # Create "empty" equivalent.
        train_dict_empty = dict.fromkeys(train_dict.keys(), ' ')

        # Copy validation statistics and add 'valid_' prefices from 
        for key, value in chkpt['validation_stats'].items():
            valid_dict['valid_{}'.format(key)] = value
        # Create "empty" equivalent.
        valid_dict_empty = dict.fromkeys(valid_dict.keys(), ' ')

        # Get all tests for a given training experiment.
        experiments_tests = self.get_experiment_tests(experiment_path)

        list_test_dicts = []
        if len(experiments_tests) > 0:
            self.logger.info('  - Found {} test(s)'.format(len(experiments_tests)))
            # "Expand" status, train and valid dicts by empty ones.
            list_status_dicts = [status_dict, *[status_dict_empty for i in range(len(experiments_tests)-1)]]
            list_train_dicts = [train_dict, *[train_dict_empty for i in range(len(experiments_tests)-1)]]
            list_valid_dicts = [valid_dict, *[valid_dict_empty for i in range(len(experiments_tests)-1)]]

            # Get tests statistics.
            for experiment_test_path in experiments_tests:
                self.logger.info('  - Analyzing test from: {}'.format(experiment_test_path))
                # Create test dict,
                test_dict = dict()
                test_dict['test_config'] = os.path.join(experiment_test_path, 'testing_set_agg_statistics.yaml')
                test_dict['test_start'] = os.path.basename(os.path.normpath(experiment_test_path))[5:]
                # Load yaml file and get random seeds.
                with open(os.path.join(experiment_test_path, 'testing_configuration.yaml'), 'r') as yaml_file:
                    test_params = yaml.load(yaml_file)  
                    # Get seeds.             
                    test_dict['test_seed_torch'] = test_params['testing']['seed_torch']
                    test_dict['test_seed_numpy'] = test_params['testing']['seed_numpy']                    

                with open(os.path.join(experiment_test_path, 'testing_statistics.csv'), mode='r') as f:
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

    def merge_list_dicts(self, list_dicts):
        """ Function merges list of ditionaries by using filling the missing fields with spaces. """

        # Create "unified" header.
        header = set(k for d in list_dicts for k in d)
        # Create "empty" dict with unified header.
        empty_dict = {k:' ' for k in header}
        # "Fill" all lists with empty gaps.
        list_filled_dicts = []
        for i,_ in enumerate(list_dicts):
            list_filled_dicts.append({**empty_dict, **(list_dicts[i])})
        # Zip lists of dics.
        final_dict = dict(zip(header, zip(*[d.values() for d in list_filled_dicts])))
        # Return the result.
        return final_dict

    def run_grid_experiment(self):
        """
        Constructor for the ``GridAnalyzer``.

        Maps the grid analysis to CPU cores in the limit of the available cores.

        """
        # Go throught experiments one by one and collect data.
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

        print(statuses)
        print(trains)
        print(valids)
        print(tests)

        # Merge everything into one big dictionary..
        exp_values =  {**statuses, **trains, **valids, **tests}
        print(exp_values)

        # create results file
        results_file = os.path.join(self.experiment_rootdir, "{0:%Y%m%d_%H%M%S}_grid_analysis.csv".format(datetime.now()))

        with open(results_file, "w") as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerow(exp_values.keys())
            writer.writerows(zip(*exp_values.values()))

        self.logger.info('Analysis done')
        self.logger.info('Results stored in {}'.format(results_file))


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
