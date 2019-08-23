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

    - Contains the definition of the ``Worker`` class, representing the base of the basic workers, such as \
    ``OnlineTrainer`` and ``Tester``.


"""
__author__ = "Vincent Marois, Tomasz Kornuta, Ryan L. McAvoy"

import os
import yaml

import torch
import logging
import logging.config
import argparse
import numpy as np
from random import randrange
from abc import abstractmethod

from torch.utils.data import DataLoader
from miprometheus.utils.sampler_factory import SamplerFactory
from miprometheus.problems.problem_factory import ProblemFactory

# Import utils.
from miprometheus.utils.app_state import AppState
from miprometheus.utils.param_interface import ParamInterface


class Worker(object):
    """
    Base abstract class for the workers.
    All base workers should subclass it and override the relevant methods.
    """

    def __init__(self, name, add_default_parser_args = True):
        """
        Base constructor for all workers:

            - Initializes the AppState singleton:

                >>> self.app_state = AppState()

            - Initializes the Parameter Registry:

                >>> self.params = ParamInterface()

            - Defines the logger:

                >>> self.logger = logging.getLogger(name=self.name)

            - Creates parser and adds default worker command line arguments.

        :param name: Name of the worker.
        :type name: str

        :param add_default_parser_args: If set, adds default parser arguments (DEFAULT: True).
        :type add_default_parser_args: bool

        """
        # Call base constructor.
        super(Worker, self).__init__()

        # Set worker name.
        self.name = name

        # Initialize the application state singleton.
        self.app_state = AppState()

        # Initialize parameter interface/registry.
        self.params = ParamInterface()

        # Initialize logger using the configuration.
        self.initialize_logger()

        # Create parser with a list of runtime arguments.
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

        # Add arguments to the specific parser.
        if add_default_parser_args:
            # These arguments will be shared by all basic workers.
            self.parser.add_argument('--config',
                                     dest='config',
                                     type=str,
                                     default='',
                                     help='Name of the configuration file(s) to be loaded. '
                                          'If specifying more than one file, they must be separated with coma ",".')

            self.parser.add_argument('--model',
                                     type=str,
                                     default='',
                                     dest='model',
                                     help='Path to the file containing the saved parameters'
                                          ' of the model to load (model checkpoint, should end with a .pt extension.)')

            self.parser.add_argument('--gpu',
                                     dest='use_gpu',
                                     action='store_true',
                                     help='The current worker will move the computations on GPU devices, if available '
                                          'in the system. (Default: False)')

            self.parser.add_argument('--expdir',
                                     dest='expdir',
                                     type=str,
                                     default="./experiments",
                                     help='Path to the directory where the experiment(s) folders are/will be stored.'
                                          ' (DEFAULT: ./experiments)')

            self.parser.add_argument('--savetag',
                                     dest='savetag',
                                     type=str,
                                     default='',
                                     help='Tag for the save directory.')

            self.parser.add_argument('--ll',
                                    action='store',
                                    dest='log_level',
                                    type=str,
                                    default='INFO',
                                    choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                                    help="Log level. (Default: INFO)")

            self.parser.add_argument('--li',
                                     dest='logging_interval',
                                     default=100,
                                     type=int,
                                     help='Statistics logging interval. Will impact logging to the logger and '
                                          'exporting to TensorBoard. Writing to the csv file is not impacted '
                                          '(interval of 1).(Default: 100, i.e. logs every 100 episodes).')

            self.parser.add_argument('--agree',
                                     dest='confirm',
                                     action='store_true',
                                     help='Request user confirmation just after loading the settings, '
                                          'before starting training. (Default: False)')

    def initialize_logger(self):
        """
        Initializes the logger, with a specific configuration:

        >>> logger_config = {'version': 1,
        >>>                  'disable_existing_loggers': False,
        >>>                  'formatters': {
        >>>                      'simple': {
        >>>                          'format': '[%(asctime)s] - %(levelname)s - %(name)s >>> %(message)s',
        >>>                          'datefmt': '%Y-%m-%d %H:%M:%S'}},
        >>>                  'handlers': {
        >>>                      'console': {
        >>>                          'class': 'logging.StreamHandler',
        >>>                          'level': 'INFO',
        >>>                          'formatter': 'simple',
        >>>                          'stream': 'ext://sys.stdout'}},
        >>>                  'root': {'level': 'DEBUG',
        >>>                           'handlers': ['console']}}

        """
        # Load the default logger configuration.
        logger_config = {'version': 1,
                         'disable_existing_loggers': False,
                         'formatters': {
                             'simple': {
                                 'format': '[%(asctime)s] - %(levelname)s - %(name)s >>> %(message)s',
                                 'datefmt': '%Y-%m-%d %H:%M:%S'}},
                         'handlers': {
                             'console': {
                                 'class': 'logging.StreamHandler',
                                 'level': 'INFO',
                                 'formatter': 'simple',
                                 'stream': 'ext://sys.stdout'}},
                         'root': {'level': 'DEBUG',
                                  'handlers': ['console']}}

        logging.config.dictConfig(logger_config)

        # Create the Logger, set its label and logging level.
        self.logger = logging.getLogger(name=self.name)

    def display_parsing_results(self):
        """
        Displays the properly & improperly parsed arguments (if any).

        """
        # Log the parsed flags.
        flags_str = 'Properly parsed command line arguments: \n'
        flags_str += '='*80 + '\n'
        for arg in vars(self.flags): 
            flags_str += "{}= {} \n".format(arg, getattr(self.flags, arg))
        flags_str += '='*80 + '\n'
        self.logger.info(flags_str)

        # Log the unparsed flags if any.
        if self.unparsed:
            flags_str = 'Invalid command line arguments: \n'
            flags_str += '='*80 + '\n'
            for arg in self.unparsed: 
                flags_str += "{} \n".format(arg)
            flags_str += '='*80 + '\n'
            self.logger.warning(flags_str)

    def setup_experiment(self):
        """
        Setups a specific experiment.

        Base method:

            - Parses command line arguments.

            - Sets the 3 default sections (training / validation / test) and sets their dataloaders params.

        .. note::

            Child classes should override this method, but still call its parent to draw the basic functionality \
            implemented here.


        """
        # Parse arguments.
        self.flags, self.unparsed = self.parser.parse_known_args()

        # Set logger depending on the settings.
        self.logger.setLevel(getattr(logging, self.flags.log_level.upper(), None))

        # add empty sections
        self.params.add_default_params({"training": {'terminal_conditions': {}}})
        self.params.add_default_params({"validation": {}})
        self.params.add_default_params({"testing": {}})

        # set a default configuration section for the DataLoaders
        dataloader_config = {'dataloader': {'shuffle': True,  # shuffle set by default.
                                            'batch_sampler': None,
                                            'num_workers': 0,  # Do not use multiprocessing by default - for now.
                                            'pin_memory': False,
                                            'drop_last': False,
                                            'timeout': 0},
                            'sampler': {},  # not using sampler by default
                            }

        self.params["training"].add_default_params(dataloader_config)
        self.params["validation"].add_default_params(dataloader_config)
        self.params["testing"].add_default_params(dataloader_config)

    def build_problem_sampler_loader(self, params, section_name):
        """
        Builds and returns the Problem class, alongside its DataLoader.

        Also builds the sampler if required.

        :param params: 'ParamInterface' object, referring to one of main sections (training/validation/testing).
        :type params: miprometheus.utils.ParamInterface

        :param section_name: name of the section that will be used by logger for display.

        :return: Problem instance & DataLoader instance.
        """

        # Build the problem.
        problem = ProblemFactory.build(params['problem'])

        # Try to build the sampler.
        sampler = SamplerFactory.build(problem, params['sampler'])

        if sampler is not None:
            # Set shuffle to False - REQUIRED as those two are exclusive.
            params['dataloader'].add_config_params({'shuffle': False})

        # build the DataLoader on top of the validation problem
        loader = DataLoader(dataset=problem,
                            batch_size=params['problem']['batch_size'],
                            shuffle=params['dataloader']['shuffle'],
                            sampler=sampler,
                            batch_sampler=params['dataloader']['batch_sampler'],
                            num_workers=params['dataloader']['num_workers'],
                            collate_fn=problem.collate_fn,
                            pin_memory=params['dataloader']['pin_memory'],
                            drop_last=params['dataloader']['drop_last'],
                            timeout=params['dataloader']['timeout'],
                            worker_init_fn=problem.worker_init_fn)

        # Display sizes.
        self.logger.info("Problem for '{}' loaded (size: {})".format(section_name, len(problem)))
        if (sampler is not None):
            self.logger.info("Sampler for '{}' created (size: {})".format(section_name, len(sampler)))


        # Return sampler - even if it is none :]
        return problem, sampler, loader


    def get_epoch_size(self, problem, sampler, batch_size, drop_last):
        """
        Compute the number of iterations ('episodes') to run given the size of the dataset and the batch size to cover
        the entire dataset once.

        Takes into account whether one used sampler or not.

        :param problem: Object derived from the ''Problem'' class

        :param sampler: Sampler (may be None)

        :param batch_size: Batch size.
        :type batch_size: int

        :param drop_last: If True then last batch (if incomplete) will not be counted
        :type drop_last: bool

        .. note::

            If the last batch is incomplete we are counting it in when ``drop_last`` in ``DataLoader()`` is set to Ttrue.

        .. warning::

            Leaving this method 'just in case', in most cases one might simply use ''len(dataloader)''.

        :return: Number of iterations to perform to go though the entire dataset once.

        """
        # "Estimate" dataset size.
        if (sampler is not None):
            problem_size = len(sampler)
        else:
            problem_size = len(problem)

        # If problem_size is a multiciplity of batch_size OR drop last is set.
        if (problem_size % batch_size) == 0 or drop_last:
            return problem_size // batch_size
        else:
            return (problem_size // batch_size) + 1


    def export_experiment_configuration(self, log_dir, filename, user_confirm):
        """
        Dumps the configuration to ``yaml`` file.

        :param log_dir: Directory used to host log files (such as the collected statistics).
        :type log_dir: str

        :param filename: Name of the ``yaml`` file to write to.
        :type filename: str

        :param user_confirm: Whether to request user confirmation.
        :type user_confirm: bool


        """
        # -> At this point, all configuration for experiment is complete.

        # Display results of parsing.
        self.display_parsing_results()

        # Log the resulting training configuration.
        conf_str = 'Final parameter registry configuration:\n'
        conf_str += '='*80 + '\n'
        conf_str += yaml.safe_dump(self.params.to_dict(), default_flow_style=False)
        conf_str += '='*80 + '\n'
        self.logger.info(conf_str)

        # Save the resulting configuration into a .yaml settings file, under log_dir
        with open(log_dir + filename, 'w') as yaml_backup_file:
            yaml.dump(self.params.to_dict(), yaml_backup_file, default_flow_style=False)

        # Ask for confirmation - optional.
        if user_confirm:
            try:
                input('Press <Enter> to confirm and start the experiment\n')
            except KeyboardInterrupt:
                exit(0)            


    def add_statistics(self, stat_col):
        """
        Adds most elementary shared statistics to ``StatisticsCollector``: episode and loss.

        :param stat_col: ``StatisticsCollector``.

        """
        # Add default statistics with formatting.
        stat_col.add_statistic('loss', '{:12.10f}')
        stat_col.add_statistic('episode', '{:06d}')

    def add_aggregators(self, stat_agg):
        """
        Adds basic statistical aggregators to ``StatisticsAggregator``: episode, \
        episodes_aggregated and loss derivatives.

        :param stat_agg: ``StatisticsAggregator``.

        """
        # add 'aggregators' for the episode.
        stat_agg.add_aggregator('episode', '{:06d}')
        # Number of aggregated episodes.
        stat_agg.add_aggregator('episodes_aggregated', '{:06d}')

        # Add default statistical aggregators for the loss (indicating a formatting).
        # Represents the average loss, but stying with loss for TensorBoard "variable compatibility".
        stat_agg.add_aggregator('loss', '{:12.10f}')
        stat_agg.add_aggregator('acc', '{:12.10f}')
        stat_agg.add_aggregator('loss_min', '{:12.10f}')
        stat_agg.add_aggregator('loss_max', '{:12.10f}')
        stat_agg.add_aggregator('loss_std', '{:12.10f}')

    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates the default statistics collected by the ``StatisticsCollector``.


        .. note::
            Only computes the min, max, mean, std of the loss as these are basic statistical aggregator by default.

            Given that the ``StatisticsAggregator`` uses the statistics collected by the ``StatisticsCollector``, \
            It should be ensured that these statistics are correctly collected (i.e. use of ``self.add_statistics()`` \
            and ``collect_statistics()``).

        :param stat_col: ``StatisticsCollector``

        :param stat_agg: ``StatisticsAggregator``

        """
        # By default, copy the last value for all variables have matching names.
        # (will work well for e.g. episode or epoch)
        for k, v in stat_col.items():
            if k in stat_agg.aggregators:
                # Copy last collected value.
                stat_agg.aggregators[k] = v[-1]

        # Get loss values.
        loss_values = stat_col['loss']
        acc_values = stat_col['acc']

        # Calculate default aggregates.
        stat_agg.aggregators['acc'] = torch.mean(torch.tensor(acc_values))
        stat_agg.aggregators['loss'] = torch.mean(torch.tensor(loss_values))
        stat_agg.aggregators['loss_min'] = min(loss_values)
        stat_agg.aggregators['loss_max'] = max(loss_values)
        stat_agg.aggregators['loss_std'] = 0.0 if len(loss_values) <= 1 else torch.std(torch.tensor(loss_values))
        stat_agg.aggregators['episodes_aggregated'] = len(loss_values)

    @abstractmethod
    def run_experiment(self):
        """
        Main function of the worker which executes a specific experiment.

        .. note::

            Abstract. Should be implemented in the subclasses.


        """

    def add_file_handler_to_logger(self, logfile):
        """
        Add a ``logging.FileHandler`` to the logger of the current ``Worker``.

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

    def recurrent_config_parse(self, configs: str, configs_parsed: list):
        """
        Parses names of configuration files in a recursive manner, i.e. \
        by looking for ``default_config`` sections and trying to load and parse those \
        files one by one.

        :param configs: String containing names of configuration files (with paths), separated by comas.
        :type configs: str

        :param configs_parsed: Configurations that were already parsed (so we won't parse them many times).
        :type configs_parsed: list


        :return: list of parsed configuration files.

        """
        # Split and remove spaces.
        configs_to_parse = configs.replace(" ", "").split(',')

        # Terminal condition.
        while len(configs_to_parse) > 0:

            # Get config.
            config = configs_to_parse.pop(0)

            # Skip empty names (after lose comas).
            if config == '':
                continue
            print("Info: Parsing the {} configuration file".format(config))

            # Check if it was already loaded.
            if config in configs_parsed:
                print('Warning: Configuration file {} already parsed - skipping'.format(config))
                continue

            # Check if file exists.
            if not os.path.isfile(config):
                print('Error: Configuration file {} does not exist'.format(config))
                exit(-1)

            try:
                # Open file and get parameter dictionary.
                with open(config, 'r') as stream:
                    param_dict = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print("Error: Couldn't properly parse the {} configuration file".format(config))
                print('yaml.YAMLERROR:', e)
                exit(-1)

            # Remember that we loaded that config.
            configs_parsed.append(config)

            # Check if there are any default configs to load.
            if 'default_configs' in param_dict:
                # If there are - recursion!
                configs_parsed = self.recurrent_config_parse(
                    param_dict['default_configs'], configs_parsed)

        # Done, return list of loaded configs.
        return configs_parsed

    def recurrent_config_load(self,configs_to_load):
        for config in reversed(configs_to_load):
            # Load params from YAML file.
            self.params.add_config_params_from_yaml(config)
            print('Loaded configuration from file {}'.format(config))

    def check_and_set_cuda(self, use_gpu):
        """
        Enables computations on CUDA if GPU is available.
        Sets the default data types.

        :param use_gpu: Command line flag indicating whether use GPU/CUDA or not. 

        """
        # Determine if GPU/CUDA is available.
        if torch.cuda.is_available():
            if use_gpu:
                self.app_state.convert_cuda_types()
                self.logger.info('Running computations on GPU using CUDA enabled')
        elif use_gpu:
            self.logger.warning('GPU flag is enabled but there are no available GPU devices, using CPU instead')
        else:
            self.logger.warning('GPU flag is disabled, using CPU.')

    def predict_evaluate_collect(self, model, problem, data_dict, stat_col, episode, epoch=None):
        """
        Function that performs the following:

            - passes samples through the model,
            - computes loss using the problem
            - collects problem and model statistics,


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
        # Collect loss as float.
        stat_col['loss'] = loss.item()

        # Collect other (potential) statistics from problem & model.
        problem.collect_statistics(stat_col, data_dict, logits)
        model.collect_statistics(stat_col, data_dict, logits)

        # Return tuple: logits, loss.
        return logits, loss

    def export_statistics(self, stat_obj, tag='', export_to_log = True):
        """
        Export the statistics/aggregations to logger, csv and TB.

        :param stat_obj: ``StatisticsCollector`` or ``StatisticsAggregato`` object.

        :param tag: Additional tag that will be added to string exported to logger, optional (DEFAULT = '').
        :type tag: str

        :param export_to_log: If True, exports statistics to logger (DEFAULT: True)
        :type export_to_log: bool

        """ 
        # Log to logger
        if export_to_log:
            self.logger.info(stat_obj.export_to_string(tag))

        # Export to csv
        stat_obj.export_to_csv()

        # Export to TensorBoard.
        stat_obj.export_to_tensorboard()

    def aggregate_and_export_statistics(self, problem, model, stat_col, stat_agg, episode, tag='', export_to_log = True):
        """
        Aggregates the collected statistics. Exports the aggregations to logger, csv and TB. \
        Empties statistics collector for the next episode.

        :param model: trainable model.
        :type model: ``models.model.Model`` or a subclass

        :param problem: problem generating samples.
        :type problem: ``problems.problem.problem`` or a subclass

        :param stat_col: ``StatisticsCollector`` object.

        :param stat_agg: ``StatisticsAggregator`` object.

        :param tag: Additional tag that will be added to string exported to logger, optional (DEFAULT = '').
        :type tag: str

        :param export_to_log: If True, exports statistics to logger (DEFAULT: True)
        :type export_to_log: bool

        """ 
        # Aggregate statistics.
        self.aggregate_statistics(stat_col, stat_agg)
        problem.aggregate_statistics(stat_col, stat_agg)
        model.aggregate_statistics(stat_col, stat_agg)

        # Set episode, so the datapoint will appear in the right place in TB.
        stat_agg["episode"] = episode

        # Export to logger, cvs and TB.
        self.export_statistics(stat_agg, tag, export_to_log)

    def cycle(self, iterable):
        """
        Cycle an iterator to prevent its exhaustion.
        This function is used in the (online) trainer to reuse the same ``DataLoader`` for a number of episodes\
        > len(dataset)/batch_size.

        :param iterable: iterable.
        :type iterable: iter

        """
        while True:
            for x in iterable:
                yield x

    def set_random_seeds(self, params, section_name):
        """
        Set ``torch`` & ``NumPy`` random seeds from the ``ParamRegistry``: \
        If one was indicated, use it, or set a random one.

        :param params: Section in config/param registry that will be changed \
            ("training" or "testing" only will be taken into account.)

        :param section_name: Name of the section (for logging purposes only).
        :type section_name: str

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
