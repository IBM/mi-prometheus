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
tester.py:

    - This file sets hosts a function which adds specific arguments a tester will need.
    - Also defines the ``Tester()`` class.


"""
__author__ = "Vincent Marois"

import os
import yaml
import torch
import argparse
import numpy as np
from time import sleep
from datetime import datetime
from torch.utils.data.dataloader import DataLoader

import workers.worker as worker
from workers.worker import Worker
from models.model_factory import ModelFactory
from problems.problem_factory import ProblemFactory
from utils.worker_utils import forward_step, handshake


def add_arguments(parser: argparse.ArgumentParser):
    """
    Add arguments to the specific parser.

    These arguments are related to the basic ``Tester``.

    :param parser: ``argparse.ArgumentParser``
    """
    # add here all arguments used by the tester.
    parser.add_argument('--model',
                        type=str,
                        default='',
                        dest='model',
                        help='Path to the file containing the saved parameters'
                             ' of the model (model checkpoint, should end with a .pt extension.)')

    parser.add_argument('--visualize',
                        action='store_true',
                        dest='visualize',
                        help='Activate dynamic visualization')


class Tester(Worker):
    """
    Defines the basic Tester.

    If defining another type of tester, it should subclass it.

    """

    def __init__(self, flags: argparse.Namespace):
        """
        Constructor for the Tester:

            - Checks that the model to use exists on file:

                >>> if not os.path.isfile(flags.model)

            - Checks that the configuration file exists:

                >>> if not os.path.isfile(config_file)

            - Set up the log directory path:

                >>> os.makedirs(self.log_dir, exist_ok=False)

            - Add a FileHandler to the logger (defined in BaseWorker):

                >>>  self.logger.addHandler(fh)

            - Set random seeds:

                >>> torch.manual_seed(self.param_interface["training"]["seed_torch"])
                >>> np.random.seed(self.param_interface["training"]["seed_numpy"])

            - Creates problem and model:

                >>> self.dataset = ProblemFactory.build_problem(self.param_interface['training']['problem'])
                >>> self.model = ModelFactory.build_model(self.param_interface['model'], self.dataset.default_values)

            - Creates the DataLoader:

                >>> self.problem = DataLoader(dataset=self.dataset, ...)


        :param flags: Parsed arguments from the parser.

        """
        # call base constructor
        super(Tester, self).__init__(flags)

        # set logger name
        self.name = 'Tester'
        self.set_logger_name(self.name)

        # delete 'epoch' entry in the StatisticsCollector as we don't need it.
        self.stat_col.__delitem__('epoch')

        # Check if model is present.
        if flags.model == '':
            print('Please pass path to and name of the file containing model to be loaded as --m parameter')
            exit(-1)

        # Check if file with model exists.
        if not os.path.isfile(flags.model):
            print('Model file {} does not exist'.format(flags.model))
            exit(-2)

        # Extract path.
        abs_path, model_dir = os.path.split(os.path.dirname(os.path.abspath(flags.model)))

        # Check if configuration file exists
        config_file = abs_path + '/training_configuration.yaml'
        if not os.path.isfile(config_file):
            print('Config file {} does not exist'.format(config_file))
            exit(-3)

        # Prepare output paths for logging
        while True:
            # Dirty fix: if log_dir already exists, wait for 1 second and try again
            try:
                time_str = 'test_{0:%Y%m%d_%H%M%S}'.format(datetime.now())
                if flags.savetag != '':
                    time_str = time_str + "_" + flags.savetag
                self.log_dir = abs_path + '/' + time_str + '/'
                os.makedirs(self.log_dir, exist_ok=False)
            except FileExistsError:
                sleep(1)
            else:
                break

        # Logging - to subdir
        self.log_file = self.log_dir + 'tester.log'

        # add the handler for the logfile to the logger
        self.add_file_handler_to_logger(self.log_file)

        if flags.visualize:
            self.app_state.visualize = True

        # Read YAML file
        self.param_interface.add_config_params_from_yaml(config_file)

        # set random seeds: reuse the ones set during training & stored in config_file (training_configuration.yaml)
        torch.manual_seed(self.param_interface["training"]["seed_torch"])
        torch.cuda.manual_seed_all(self.param_interface["training"]["seed_torch"])
        self.logger.info('Reusing training seed_torch: {}'.format(self.param_interface["training"]["seed_torch"]))

        np.random.seed(self.param_interface["training"]["seed_numpy"])
        self.logger.info('Reusing training seed_numpy: {}'.format(self.param_interface["training"]["seed_numpy"]))

        # check if CUDA is available turn it on
        check_and_set_cuda(self.param_interface['testing'], self.logger)

        # Get problem and model names.
        try:
            _ = self.param_interface['testing']['problem']['name']
        except KeyError:
            print("Error: Couldn't retrieve the problem name from the loaded configuration")
            exit(-1)

        try:
            _ = self.param_interface['model']['name']
        except KeyError:
            print("Error: Couldn't retrieve model name from the loaded configuration")
            exit(-1)

        # Create model object.
        self.model = ModelFactory.build_model(self.param_interface['model'])
        self.model.cuda() if self.app_state.use_CUDA else None

        # Load parameters from checkpoint.
        self.model.load(flags.model)

        # Turn on evaluation mode.
        self.model.eval()

        # Build problem.
        self.problem = ProblemFactory.build_problem(self.param_interface['testing']['problem'])

        # perform 2-way handshake between Model and Problem
        handshake(model=self.model, problem=self.problem, logger=self.logger)
        # no error thrown, so handshake succeeded

        # build the DataLoader on top of the Problem class
        self.dataloader = DataLoader(dataset=self.problem,
                                     batch_size=self.param_interface['testing']['problem']['batch_size'],
                                     shuffle=self.param_interface['testing']['dataloader']['shuffle'],
                                     sampler=self.param_interface['testing']['dataloader']['sampler'],
                                     batch_sampler=self.param_interface['testing']['dataloader']['batch_sampler'],
                                     num_workers=self.param_interface['testing']['dataloader']['num_workers'],
                                     collate_fn=self.problem.collate_fn,
                                     pin_memory=self.param_interface['testing']['dataloader']['pin_memory'],
                                     drop_last=self.param_interface['testing']['dataloader']['drop_last'],
                                     timeout=self.param_interface['testing']['dataloader']['timeout'],
                                     worker_init_fn=self.problem.worker_init_fn)

        # check if the maximum number of episodes is specified, if not put a
        # default equal to the size of the dataset (divided by the batch size)
        # So that by default, we loop over the test set once.
        max_test_episodes = self.problem.get_epoch_size(self.param_interface['testing']['problem']['batch_size'])

        if "max_test_episodes" not in self.param_interface["testing"]["problem"] \
                or self.param_interface["testing"]["problem"]["max_test_episodes"] == -1:
            self.param_interface['testing']['problem'].add_config_params({'max_test_episodes': max_test_episodes})

        # warn if indicated number of episodes is larger than an epoch size:
        if self.param_interface["testing"]["problem"]["max_test_episodes"] > max_test_episodes:
            self.logger.warning('Indicated maximum number of episodes is larger than one epoch, reducing it.')
            self.param_interface['testing']['problem'].add_config_params({'max_test_episodes': max_test_episodes})

        self.logger.info("Setting the max number of episodes to: {}".format(
            self.param_interface["testing"]["problem"]["max_test_episodes"]))

        # Add model/problem dependent statistics.
        self.problem.add_statistics(self.stat_col)
        self.model.add_statistics(self.stat_col)

        # Create test output csv file.
        self.test_file = self.stat_col.initialize_csv_file(self.log_dir, 'testing.csv')

        # Ok, finished loading the configuration.
        # Save the resulting configuration into a yaml settings file, under log_dir
        with open(self.log_dir + "testing_configuration.yaml", 'w') as yaml_backup_file:
            yaml.dump(self.param_interface.to_dict(),
                      yaml_backup_file, default_flow_style=False)

    def forward(self, flags: argparse.Namespace):
        """
        Main function of the ``Tester``: Test the loaded model over the test set.

        Iterates over the ``DataLoader`` for a maximum number of episodes equal to the test set size.

        The function does the following for each episode:

            - Forwards pass of the model,
            - Logs statistics & accumulates loss,
            - Activate visualization if set.

        """

        # Ask for confirmation - optional.
        if flags.confirm:
            input('Press any key to continue')

        # Run test
        with torch.no_grad():

            acc_loss = 0
            episode = 0
            for data_dict in self.dataloader:

                if episode == self.param_interface["testing"]["problem"]["max_test_episodes"]:
                    break

                logits, loss = forward_step(self.model, self.problem, episode, self.stat_col, data_dict)
                acc_loss += loss

                # Log to logger.
                self.logger.info(self.stat_col.export_statistics_to_string('[Test]'))
                # Export to csv.
                self.stat_col.export_statistics_to_csv(self.test_file)

                if self.app_state.visualize:

                    # Allow for preprocessing
                    data_dict, logits = self.problem.plot_preprocessing(data_dict, logits)

                    # Show plot, if user presses Quit - break.
                    is_closed = self.model.plot(data_dict, logits)
                    if is_closed:
                        break

                # move to next episode.
                episode += 1

            self.logger.info('Test finished!')

            # TODO: move to StatisticsAggregator for this.
            self.logger.info('Average loss over the test set: {}'.format(acc_loss/episode))


if __name__ == '__main__':
    # Create parser with list of  runtime arguments.
    argp = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # add default arguments
    worker.add_arguments(argp)

    # add trainers-specific arguments
    add_arguments(argp)

    # Parse arguments.
    FLAGS, unparsed = argp.parse_known_args()

    tester = Tester(FLAGS)
    tester.forward(FLAGS)
