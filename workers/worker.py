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
worker.py:

    - This file contains the base parser used by every worker. Having a root common parser\
     shared by all workers allow easier future maintenance, as it eases adding an argument\
      to all workers. Each worker is able to add specific arguments using the ``add_arguments()``\
       function defined in the relevant module.

    - The base worker class is also defined here.


"""
__author__ = "Vincent Marois, Tomasz Kornut, Ryan L. McAvoy"

import yaml
import torch
import logging
import argparse
import numpy as np
import logging.config
from random import randrange
from abc import abstractmethod

# Import utils.
from utils.app_state import AppState
from utils.param_interface import ParamInterface
from utils.statistics_collector import StatisticsCollector
from utils.statistics_aggregator import StatisticsAggregator


def add_arguments(parser: argparse.ArgumentParser):
    """
    Add arguments to the specific parser.
    These arguments will be shared by all workers.
    :param parser: ``argparse.ArgumentParser``
    """
    #  It is possible to add arguments to all workers by adding them in this function.

    parser.add_argument('--savetag',
                        dest='savetag',
                        type=str,
                        default='',
                        help='Tag for the save directory')

    parser.add_argument('--log',
                        action='store',
                        dest='log',
                        type=str,
                        default='INFO',
                        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                        help="Log level. (Default: INFO)")

    # add here all arguments used by the trainers.
    parser.add_argument('--agree',
                        dest='confirm',
                        action='store_true',
                        help='Request user confirmation just after loading the settings, '
                             'before starting training  (Default: False)')


class Worker(object):
    """
    Base abstract class for the workers.
    All workers should subclass it and override the relevant methods.
    """

    def __init__(self, flags: argparse.Namespace):
        """
        Base constructor for all workers:

            - Initializes the Parameter Registry:

                >>> self.params = ParamInterface()

            - Defines the logger:

                >>> self.logger = logging.getLogger(name=self.name)

            - Initialize the AppState singleton:

                >>> self.app_state = AppState()

            - Create the StatisticsCollector:

                >>> self.stat_col = StatisticsCollector()

            - Create the StatisticsAggregator:

                >>> self.stat_agg = StatisticsAggregator()



        :param flags: Parsed arguments from the parser.

        """
        # call base constructor
        super(Worker, self).__init__()

        # default name
        self.name = 'Worker'

        # Initialize parameter interface.
        self.params = ParamInterface()

        # add empty sections
        self.params.add_default_params({"training": {}})
        self.params.add_default_params({"validation": {}})
        self.params.add_default_params({"testing": {}})

        # set a default configuration section for the DataLoaders
        dataloader_config = {'dataloader': {'shuffle': False,
                                            'sampler': None,
                                            'batch_sampler': None,
                                            'num_workers': 0,  # use multiprocessing by default
                                            'pin_memory': False,
                                            'drop_last': False,
                                            'timeout': 0}}

        self.params["training"].add_default_params(dataloader_config)
        self.params["validation"].add_default_params(dataloader_config)
        self.params["testing"].add_default_params(dataloader_config)

        # Load the default logger configuration.
        with open('logger_config.yaml', 'rt') as f:
            config = yaml.load(f.read())
            logging.config.dictConfig(config)

        # Create the Logger, set its label and logging level.
        self.logger = logging.getLogger(name=self.name)
        self.logger.setLevel(getattr(logging, flags.log.upper(), None))

        # Initialize the application state singleton.
        self.app_state = AppState()


    @abstractmethod
    def forward(self, flags: argparse.Namespace):
        """
        Main function of the worker which executes a specific task.

        .. note::

            Abstract. Should be implemented in the subclasses.

        :param flags: Parsed arguments from the command line.

        """

    def check_and_set_cuda(self, params):
        """
        Enables CUDA if available and sets the default data types.

        :param params: Section in config/param registry that will be used \
            ("training" or "test" will have effect only!)

        """
        # Set CUDA to false by default.
        params.add_default_params({'cuda': False})
        # Get actual CUDA value.
        turn_on_cuda = params['cuda']

        # Determine if CUDA is to be used.
        if torch.cuda.is_available():
            if turn_on_cuda:
                self.app_state.convert_cuda_types()
                self.logger.info('Running with CUDA enabled')
        elif turn_on_cuda:
            self.logger.error('CUDA is enabled but there is no available device')


    def predict_evaluate_collect(self, model, problem, data_dict, stat_col, episode, epoch=None):
        """
        Function that performs the following:

            - passes samples through the model,
            - calculates loss using the problem
            - collect problem and model statistics,

        :param model: trainable model.
        :type model: ``models.model.Model`` or a subclass

        :param problem: problem generating samples.
        :type problem: ``problems.problem.problem`` or a subclass

        :param data_dict: contains the batch of samples to pass to the model.
        :type data_dict: ``DataDict``

        :param stat_col: statistics collector used for logging accuracy etc.
        :type stat_col: ``StatisticsCollector``

        :param episode: current episode index
        :type episode: int

        :param epoch: current epoch index.
        :type epoch: int, optional


        :return:

            - logits,
            - loss

        """
        # Convert to CUDA.
        if self.app_state.use_CUDA:
            data_dict = data_dict.cuda()

        # Perform forward calculation.
        logits = model(data_dict)

        # Evaluate loss function.
        loss = problem.evaluate_loss(data_dict, logits)

        # Collect "elementary" statistics - episode and loss.
        if ('epoch' in stat_col) and (epoch is not None):
            stat_col['epoch'] = epoch

        stat_col['episode'] = episode
        stat_col['loss'] = loss

        # Collect other (potential) statistics from problem & model.
        problem.collect_statistics(stat_col, data_dict, logits)
        model.collect_statistics(stat_col, data_dict, logits)

        # Return tuple: logits, loss.
        return logits, loss


    def cycle(self, iterable):
        """
        Cycle an iterator to prevent its exhaustion.
        This function is used in the (episodic) trainer to reuse the same ``DataLoader`` for a number of episodes\
        > len(dataset)/batch_size.

        :param iterable: iterable.
        :type iterable: iter

        """
        while True:
            for x in iterable:
                yield x


    def set_logger_name(self, name):
        """
        Set name of ``self.logger``.

        :param name: New name for the ``logging.Logger``.
        :type name: str

        """
        self.logger.name = name

    def add_file_handler_to_logger(self, logfile):
        """
        Add a ``logging.FileHandler`` to the logger of the current Worker.

        Specifies a ``logging.Formatter``:

            >>> logging.Formatter(fmt='[%(asctime)s] - %(levelname)s - %(name)s >>> %(message)s',
            >>>                   datefmt='%Y-%m-%d %H:%M:%S')


        :param logfile: File used by the ``FileHandler``.

        """
        # create file handler which logs even DEBUG messages
        fh = logging.FileHandler(logfile)

        # set logging level for this file
        fh.setLevel(logging.DEBUG)

        # create formatter and add it to the handlers
        formatter = logging.Formatter(fmt='[%(asctime)s] - %(levelname)s - %(name)s >>> %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)

        # add the handler to the logger
        self.logger.addHandler(fh)

    def set_random_seeds(self, params, section_name):
        """
        Set ``torch`` & ``NumPy`` random seeds from the ParamRegistry:\
        If one was indicated, use it, or set a random one.

        :param params: Section in config/param registry that will be changed \
            ("training" or "test" will have effect only!)

        :param section_name: Name of the section (for logging purposes only)

        """
        # Set the random seeds: either from the loaded configuration or a default randomly selected one.
        params.add_default_params({"seed_numpy": -1})
        if params["seed_numpy"] == -1:
            seed = randrange(0, 2 ** 32)
            # Overwrite the config param!
            params.add_config_params({"seed_numpy": seed})

        self.logger.info("Setting numpy random seed in {} to: {}".format(section_name, params["seed_numpy"]))
        np.random.seed(params["seed_numpy"])

        params.add_default_params({"seed_torch": -1})
        if params["seed_torch"] == -1:
            seed = randrange(0, 2 ** 32)
            # Overwrite the config param!
            params.add_config_params({"seed_torch": seed})

        self.logger.info("Setting torch random seed in {} to: {}".format(section_name, params["seed_torch"]))
        torch.manual_seed(params["seed_torch"])
        torch.cuda.manual_seed_all(params["seed_torch"])