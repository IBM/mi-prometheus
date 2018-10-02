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
base_tester.py:

    - This file sets hosts a function which adds specific arguments a tester will need.
    - Also defines the Tester() class.


"""
__author__ = "Vincent Marois"

import os
import yaml
import torch
import logging
import argparse
import numpy as np
from time import sleep
from random import randrange
from datetime import datetime
from torch.utils.data.dataloader import DataLoader

import workers.base_worker as worker
from workers.base_worker import BaseWorker
from models.model_factory import ModelFactory
from problems.problem_factory import ProblemFactory
from utils.worker_utils import forward_step, check_and_set_cuda, handshake


def add_arguments(parser: argparse.ArgumentParser):
    """
    Add arguments to the specific parser.
    These arguments are related to the basic Tester.
    :param parser: ``argparse.ArgumentParser``
    """
    # add here all arguments used by the tester.
    parser.add_argument('--model',
                        type=str,
                        default='',
                        dest='model',
                        help='Path to and name of the file containing the saved parameters'
                             ' of the model (model checkpoint)')

    parser.add_argument('--visualize',
                        action='store_true',
                        dest='visualize',
                        help='Activate dynamic visualization')


class Tester(BaseWorker):
    """
    Defines the basic Tester.

    TODO: Enhance documentation.
    """

    def __init__(self, flags: argparse.Namespace):
        """
        Constructor for the Tester:

            -

        :param flags: Parsed arguments from the parser.

        TODO: Enhance documentation
        """
        # default name (define it before calling base constructor for logger)
        self.name = 'Tester'

        # call base constructor
        super(Tester, self).__init__(flags)

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

        # the logger is created in BaseWorker.__init__(), now we need to add to add the handler for the logfile
        # create file handler which logs even DEBUG messages
        fh = logging.FileHandler(self.log_file)
        # set logging level for this file
        fh.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(fmt='[%(asctime)s] - %(levelname)s - %(name)s >>> %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        # add the handler to the logger
        self.logger.addHandler(fh)

        if flags.visualize:
            self.app_state.visualize = True

        # Read YAML file
        with open(config_file, 'r') as stream:
            self.param_interface.add_custom_params(yaml.load(stream))

        # Set random seeds.
        if "seed_torch" not in self.param_interface["testing"] or self.param_interface["testing"]["seed_torch"] == -1:
            seed = randrange(0, 2 ** 32)
            self.param_interface["testing"].add_custom_params({"seed_torch": seed})
        self.logger.info("Setting torch random seed to: {}".format(self.param_interface["testing"]["seed_torch"]))
        torch.manual_seed(self.param_interface["testing"]["seed_torch"])
        torch.cuda.manual_seed_all(self.param_interface["testing"]["seed_torch"])

        if "seed_numpy" not in self.param_interface["testing"] or self.param_interface["testing"]["seed_numpy"] == -1:
            seed = randrange(0, 2 ** 32)
            self.param_interface["testing"].add_custom_params({"seed_numpy": seed})
        self.logger.info("Setting numpy random seed to: {}".format(self.param_interface["testing"]["seed_numpy"]))
        np.random.seed(self.param_interface["testing"]["seed_numpy"])

        # check if CUDA is available turn it on
        check_and_set_cuda(self.param_interface['testing'], self.logger)

        # Get problem and model names.
        try:
            task_name = self.param_interface['testing']['problem']['name']
        except BaseException:
            print("Error: Couldn't retrieve the problem name from the loaded configuration")
            exit(-1)

        try:
            model_name = self.param_interface['model']['name']
        except BaseException:
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
        self.dataset = ProblemFactory.build_problem(self.param_interface['testing']['problem'])

        # perform 2-way handshake between Model and Problem
        handshake(model=self.model, problem=self.dataset, logger=self.logger)
        # no error thrown, so handshake succeeded

        # build the DataLoader on top of the Problem class
        # Set a default number of workers to 4
        # TODO: allow the user to change the num_workers and other attributes value of the DataLoader.
        self.problem = DataLoader(dataset=self.dataset,
                                  batch_size=self.param_interface['testing']['problem']['batch_size'],
                                  shuffle=True,
                                  collate_fn=self.dataset.collate_fn,
                                  num_workers=4,
                                  worker_init_fn=self.dataset.worker_init_fn,
                                  drop_last=False)

        # check if the maximum number of episodes is specified, if not put a
        # default equal to the size of the dataset (divided by the batch size)
        if "max_test_episodes" not in self.param_interface["testing"]["problem"] \
                or self.param_interface["testing"]["problem"]["max_test_episodes"] == -1:
            max_test_episodes = self.dataset.get_epoch_size(self.param_interface['training']['problem']['batch_size'])

            self.param_interface['testing']['problem'].add_custom_params({'max_test_episodes': max_test_episodes})

        self.logger.info("Setting the max number of episodes to: {}".format(
            self.param_interface["testing"]["problem"]["max_test_episodes"]))

        # Add model/problem dependent statistics.
        self.dataset.add_statistics(self.stat_col)
        self.model.add_statistics(self.stat_col)

        # Create test output csv file.
        self.test_file = self.stat_col.initialize_csv_file(self.log_dir, 'testing.csv')

        # Ok, finished loading the configuration.
        # Save the resulting configuration into a yaml settings file, under log_dir
        with open(self.log_dir + "testing_configuration.yaml", 'w') as yaml_backup_file:
            yaml.dump(self.param_interface.to_dict(),
                      yaml_backup_file, default_flow_style=False)

    def forward(self):
        """
        Runs a test.

        TODO: Enhance documentation.
        """

        # Run test
        with torch.no_grad():
            acc_loss = 0

            for episode, data_dict in enumerate(self.problem):

                if episode == self.param_interface["testing"]["problem"]["max_test_episodes"]:
                    break

                logits, loss = forward_step(self.model, self.dataset, episode, self.stat_col, data_dict)
                acc_loss += loss

                # Log to logger.
                self.logger.info(self.stat_col.export_statistics_to_string('[Test]'))
                # Export to csv.
                self.stat_col.export_statistics_to_csv(self.test_file)

                if self.app_state.visualize:

                    # Allow for preprocessing
                    data_dict, logits = self.dataset.plot_preprocessing(data_dict, logits)

                    # Show plot, if user presses Quit - break.
                    is_closed = self.model.plot(data_dict, logits)
                    if is_closed:
                        break
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
    tester.forward()
