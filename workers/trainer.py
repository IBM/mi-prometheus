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
trainer.py:

    - This file sets hosts a function which adds specific arguments a trainer will need.
    - Also defines the ``Trainer()`` class, which is the default, epoch-based trainer.


"""
__author__ = "Vincent Marois, Tomasz Kornuta"

import os
import yaml
import torch
import argparse
import numpy as np
from random import randrange
from time import sleep
from datetime import datetime
from torch.nn.utils import clip_grad_value_
from torch.utils.data.dataloader import DataLoader


import workers.worker as worker
from workers.worker import Worker
from models.model_factory import ModelFactory
from problems.problem_factory import ProblemFactory

from utils.worker_utils import recurrent_config_parse, handshake


def add_arguments(parser: argparse.ArgumentParser):
    """
    Add arguments to the specific parser.
    These arguments will be shared by all (basic) trainers.
    :param parser: ``argparse.ArgumentParser``
    """
    parser.add_argument('--config',
                        dest='config',
                        type=str,
                        default='',
                        help='Name of the configuration file(s) to be loaded.'
                             'If specifying more than one file, they must be separated with coma ",".)')

    parser.add_argument('--outdir',
                        dest='outdir',
                        type=str,
                        default="./experiments",
                        help='Path to the output directory where the experiment(s) folders will be stored.'
                             ' (DEFAULT: ./experiments)')

    parser.add_argument('--model',
                        type=str,
                        default='',
                        dest='model',
                        help='Path to the file containing the saved parameters'
                             ' of the model to load (model checkpoint, should end with a .pt extension.)')

    parser.add_argument('--tensorboard',
                        action='store',
                        dest='tensorboard', choices=[0, 1, 2],
                        type=int,
                        help="If present, enable logging to TensorBoard. Available log levels:\n"
                             "0: Log the collected statistics.\n"
                             "1: Add the histograms of the model's biases & weights (Warning: Slow).\n"
                             "2: Add the histograms of the model's biases & weights gradients (Warning: Even slower).")

    parser.add_argument('--lf',
                        dest='logging_frequency',
                        default=100,
                        type=int,
                        help='Statistics logging frequency. Will impact logging to the logger and exporting to '
                             'TensorBoard. Writing to the csv file is not impacted (frequency of 1).'
                             '(Default: 100, i.e. logs every 100 episodes).')

    parser.add_argument('--visualize',
                        dest='visualize',
                        default='-1',
                        choices=[-1, 0, 1, 2, 3],
                        type=int,
                        help="Activate dynamic visualization (Warning: will require user interaction):\n"
                             "-1: disabled (DEFAULT)\n"
                             "0: Only during training episodes.\n"
                             "1: During both training and validation episodes.\n"
                             "2: Only during validation episodes.\n"
                             "3: Only during the last validation, after the training is completed.\n")


class Trainer(Worker):
    """
    Base class for the trainers.

    Iterates over epochs on the dataset.

    All other types of trainers (e.g. ``EpisodeTrainer``) should subclass it.

    """

    def __init__(self, flags: argparse.Namespace):
        """
        Base constructor for all trainers:

            - Loads the config file(s):

                >>> configs_to_load = recurrent_config_parse(flags.config, [])

            - Set up the log directory path:

                >>> os.makedirs(self.log_dir, exist_ok=False)

            - Add a FileHandler to the logger (defined in BaseWorker):

                >>>  self.add_file_handler_to_logger(self.log_file)

            - Handles TensorBoard writers & files:

                >>> self.training_writer = SummaryWriter(self.log_dir + '/training')

            - Set random seeds:

                >>> torch.manual_seed(self.params["training"]["seed_torch"])
                >>> np.random.seed(self.params["training"]["seed_numpy"])

            - Creates problem and model:

                >>> self.dataset = ProblemFactory.build_problem(self.params['training']['problem'])
                >>> self.model = ModelFactory.build_model(self.params['model'], self.dataset.default_values)

            - Creates the DataLoader:

                >>> self.problem = DataLoader(dataset=self.problem, ...)

            - Handles curriculum learning if indicated:

                >>> if 'curriculum_learning' in self.params['training']:
                >>> ...

            - Handles the validation of the model:

                - Instantiates the problem class, with the parameters contained in the `validation` section,
                - Will validate the model at the end of each epoch, over the entire validation set, and log the \
                statistical aggregators (minimum / maximum / average / standard deviation... of the loss, accuracy \
                etc.), \
                - Will validate the model again at the end of training if one of the terminal conditions is met.


            - Set optimizer:

                >>> self.optimizer = getattr(torch.optim, optimizer_name)


        :param flags: Parsed arguments from the parser.

        """
        # call base constructor
        super(Trainer, self).__init__(flags)

        # set name of logger
        self.name = 'Trainer'
        self.set_logger_name(self.name)

        # Check if config file was selected.
        if flags.config == '':
            print('Please pass configuration file(s) as --c parameter')
            exit(-1)

        # Get the list of configurations which need to be loaded.
        configs_to_load = recurrent_config_parse(flags.config, [])

        # Read the YAML files one by one - but in reverse order -> overwrite the first indicated config(s)
        for config in reversed(configs_to_load):
            # Load params from YAML file.
            self.params.add_config_params_from_yaml(config)
            print('Loaded configuration from file {}'.format(config))

        # -> At this point, the Param Registry contains the configuration loaded (and overwritten) from several files.

        # Get training problem name.
        try:
            training_problem_name = self.params['training']['problem']['name']
        except KeyError:
            print("Error: Couldn't retrieve problem name from the 'training' section in the loaded configuration")
            exit(-1)

        # Get validation problem name
        try:
            _ = self.params['validation']['problem']['name']
        except KeyError:
            print("Error: Couldn't retrieve problem name from the 'validation' section in the loaded configuration")
            exit(-1)

        # Get model name.
        try:
            model_name = self.params['model']['name']
        except KeyError:
            print("Error: Couldn't retrieve model name from the loaded configuration")
            exit(-1)

        # Prepare the output path for logging
        while True:  # Dirty fix: if log_dir already exists, wait for 1 second and try again
            try:
                time_str = '{0:%Y%m%d_%H%M%S}'.format(datetime.now())
                if flags.savetag != '':
                    time_str = time_str + "_" + flags.savetag
                self.log_dir = flags.outdir + '/' + training_problem_name + '/' + model_name + '/' + time_str + '/'
                os.makedirs(self.log_dir, exist_ok=False)
            except FileExistsError:
                sleep(1)
            else:
                break

        self.model_dir = self.log_dir + 'models/'
        os.makedirs(self.model_dir, exist_ok=False)

        # add the handler for the logfile to the logger
        self.log_file = self.log_dir + 'trainer.log'
        self.add_file_handler_to_logger(self.log_file)

        # Set random seeds in the training section.
        self.set_random_seeds(self.params['training'], 'training')

        # check if CUDA is available, if yes turn it on
        self.check_and_set_cuda(self.params['training'])

        # Build the problem for the training
        self.problem = ProblemFactory.build_problem(self.params['training']['problem'])

        # check that the number of epochs is available in param_interface. If not, put a default of 1.
        if "max_epochs" not in self.params["training"]["terminal_condition"] \
                or self.params["training"]["terminal_condition"]["max_epochs"] == -1:
            max_epochs = 1

            self.params["training"]["terminal_condition"].add_config_params({'max_epochs': max_epochs})

        self.logger.info("Setting the max number of epochs to: {}".format(
            self.params["training"]["terminal_condition"]["max_epochs"]))

        # ge t theepoch size in terms of episodes:
        epoch_size = self.problem.get_epoch_size(self.params["training"]["problem"]["batch_size"])
        self.logger.info('Epoch size in terms of episodes: {}'.format(epoch_size))

        # Build the model using the loaded configuration and the default values of the problem.
        self.model = ModelFactory.build_model(self.params['model'], self.problem.default_values)

        # load the indicated pretrained model checkpoint if the argument is valid
        if flags.model != "":
            if os.path.isfile(flags.model):
                # Load parameters from checkpoint.
                self.model.load(flags.model)
            else:
                self.logger.error("Couldn't load the checkpoint {} : does not exist on disk.".format(flags.model))

        # move the model to CUDA if applicable
        if self.app_state.use_CUDA:
            self.model.cuda()

        # perform 2-way handshake between Model and Problem
        handshake(model=self.model, problem=self.problem, logger=self.logger)
        # no error thrown, so handshake succeeded

        # Log the model summary.
        self.logger.info(self.model.summarize())

        # build the DataLoader on top of the Problem class, using the associated configuration section.
        self.dataloader = DataLoader(dataset=self.problem,
                                     batch_size=self.params['training']['problem']['batch_size'],
                                     shuffle=self.params['training']['dataloader']['shuffle'],
                                     sampler=self.params['training']['dataloader']['sampler'],
                                     batch_sampler=self.params['training']['dataloader']['batch_sampler'],
                                     num_workers=self.params['training']['dataloader']['num_workers'],
                                     collate_fn=self.problem.collate_fn,
                                     pin_memory=self.params['training']['dataloader']['pin_memory'],
                                     drop_last=self.params['training']['dataloader']['drop_last'],
                                     timeout=self.params['training']['dataloader']['timeout'],
                                     worker_init_fn=self.problem.worker_init_fn)

        # parse the curriculum learning section in the loaded configuration.
        if 'curriculum_learning' in self.params['training']:

            # Initialize curriculum learning - with values from loaded configuration.
            self.problem.curriculum_learning_initialize(self.params['training']['curriculum_learning'])

            # Set initial values of curriculum learning.
            self.curric_done = self.problem.curriculum_learning_update_params(0)

            # If the 'must_finish' key is not present in config then then it will be finished by default
            if 'must_finish' not in self.params['training']['curriculum_learning']:
                self.params['training']['curriculum_learning'].add_default_params({'must_finish': True})

            self.must_finish_curriculum = self.params['training']['curriculum_learning']['must_finish']
            self.logger.info("Using curriculum learning")

        else:
            # Initialize curriculum learning - with empty dict.
            self.problem.curriculum_learning_initialize({})

            # If not using curriculum learning then it does not have to be finished.
            self.must_finish_curriculum = False

        # Build the validation problem.
        self.validation_problem = ProblemFactory.build_problem(self.params['validation']['problem'])

        # build the DataLoader on top of the validation problem
        self.validation_dataloader = DataLoader(dataset=self.validation_problem,
                                   batch_size=self.params['validation']['problem']['batch_size'],
                                   shuffle=self.params['validation']['dataloader']['shuffle'],
                                   sampler=self.params['validation']['dataloader']['sampler'],
                                   batch_sampler=self.params['validation']['dataloader']['batch_sampler'],
                                   num_workers=self.params['validation']['dataloader']['num_workers'],
                                   collate_fn=self.validation_problem.collate_fn,
                                   pin_memory=self.params['validation']['dataloader']['pin_memory'],
                                   drop_last=self.params['validation']['dataloader']['drop_last'],
                                   timeout=self.params['validation']['dataloader']['timeout'],
                                   worker_init_fn=self.validation_problem.worker_init_fn)

        # Set the optimizer.
        optimizer_conf = dict(self.params['training']['optimizer'])
        optimizer_name = optimizer_conf['name']
        del optimizer_conf['name']

        # Instantiate the optimizer and filter the model parameters based on if they require gradients.
        self.optimizer = getattr(torch.optim, optimizer_name)(filter(lambda p: p.requires_grad,
                                                                     self.model.parameters()),
                                                              **optimizer_conf)

        # -> At this point, all configuration for the ``Trainer`` is complete.

        # Add the model & problem dependent statistics to the ``StatisticsCollector``
        self.problem.add_statistics(self.stat_col)
        self.model.add_statistics(self.stat_col)

        # Add the model & problem dependent statistical aggregators to the ``StatisticsEstimators``
        self.problem.add_aggregators(self.stat_agg)
        self.model.add_aggregators(self.stat_agg)

        # Save the resulting configuration into a .yaml settings file, under log_dir
        with open(self.log_dir + "training_configuration.yaml", 'w') as yaml_backup_file:
            yaml.dump(self.params.to_dict(), yaml_backup_file, default_flow_style=False)

        # Log the resulting training configuration.
        conf_str = '\n' + '='*80 + '\n'
        conf_str += 'Final registry configuration for training {} on {}:\n'.format(model_name, training_problem_name)
        conf_str += '='*80 + '\n'
        conf_str += yaml.safe_dump(self.params.to_dict(), default_flow_style=False)
        conf_str += '='*80 + '\n'
        self.logger.info(conf_str)


    def initialize_tensorboard(self, tensorboard_flag):
        """
        Function initializes tensorboard

        :param tensorboard_flag: Flag set from command line. If not None, it will activate different \
            modes of TB summary writer

        """
        # Create TensorBoard outputs - if TensorBoard is supposed to be used.
        if tensorboard_flag is not None:
            from tensorboardX import SummaryWriter

            self.training_writer = SummaryWriter(self.log_dir + '/training')
            self.validation_writer = SummaryWriter(self.log_dir + '/validation')
        else:
            self.training_writer = None
            self.validation_writer = None


    def finalize_tensorboard(self):
        """ 
        Finalizes operation of TensorBoard writers.
        """
        # Close the TensorBoard writers.
        if self.training_writer is not None:
            self.training_writer.close()
        if self.validation_writer is not None:
            self.validation_writer.close()
        

    def initialize_statistics_collection(self):
        """
        Function initializes all statistics collectors and aggregators used by a given worker,
        creates output files etc.
        """
        # Add statistics characteristic for this (i.e. epoch) trainer.
        self.stat_col.add_statistic('epoch', '{:06d}')
        self.stat_agg.add_aggregator('epoch', '{:06d}')

        # Create the csv file to store the training statistics.
        self.training_stats_file = self.stat_col.initialize_csv_file(self.log_dir, 'training_statistics.csv')

        # Create the csv file to store the training statistical estimators.
        # doing it in the forward, not constructor, as the ``EpisodicTrainer`` does not need it.
        self.training_stats_aggregated_file = self.stat_agg.initialize_csv_file(self.log_dir, 'training_aggregated_statistics.csv')

        # Create the csv file to store the validation statistical aggregators
        # This file will contains several data points for the ``Trainer`` (but only one for the ``EpisodicTrainer``)
        self.validation_stats_aggregated_file = self.stat_agg.initialize_csv_file(self.log_dir, 'validation_aggregated_statistics.csv')


    def finalize_statistics_collection(self):
        """
        Finalizes statistics collection, closes all files etc.
        """
        # Close all files.
        self.training_stats_file.close()
        self.training_stats_aggregated_file.close()
        self.validation_stats_aggregated_file.close()


    def validation_step(self, valid_batch, episode, epoch=None):
        """
        Performs a validation step on the model, using the provided data batch.

        Additionally logs results (to files, tensorboard) and handles visualization.

        :param valid_batch: data batch generated by the problem and used as input to the model.
        :type valid_batch: ``DataDict``

        :param stat_col: statistics collector used for logging accuracy etc.
        :type stat_col: ``StatisticsCollector``

        :param episode: current training episode index.
        :type episode: int

        :param epoch: current epoch index.
        :type epoch: int, optional

        :return:

            - Validation loss,
            - if AppState().visualize:
                return True if the user closed the window, else False
            else:
                return False, i.e. continue training.

        """
        # Turn on evaluation mode.
        self.model.eval()

        # Compute the validation loss using the provided data batch.
        with torch.no_grad():
            valid_logits, valid_loss = self.predict_and_evaluate(self.model, self.validation_problem, valid_batch, episode, epoch)

        # Log to logger.
        self.logger.info(self.stat_col.export_statistics_to_string('[Validation on a single batch]'))

        # Export to csv.
        self.stat_col.export_statistics_to_csv(self.validation_stats_file)

        if self.validation_writer is not None:
            # Save loss + accuracy to TensorBoard.
            self.stat_col.export_statistics_to_tensorboard(self.validation_writer)

        # Visualization of validation.
        if self.app_state.visualize:
            # Allow for preprocessing
            valid_batch, valid_logits = self.problem.plot_preprocessing(valid_batch, valid_logits)

            # Show plot, if user will press Stop then a SystemExit exception will be thrown.
            self.model.plot(valid_batch, valid_logits)

        # Else simply return false, i.e. continue training.
        return valid_loss


    def validation_over_set(self, episode, epoch=None):
        """
        Performs a validation of the model on the whole validation set, using the validation dataloader.

        Iterates over the entire validation set (through the dataloader), aggregates the collected statistics\ 
        and logs that to the console, csv and tensorboard (if set).

        If visualization is activated, this function will select a random batch to visualize.

        :param episode: current training episode index.
        :type episode: int

        :param epoch: current epoch index.
        :type epoch: int, optional

        :return:

            - Average loss over the validation set.
            - if ``AppState().visualize``:
                return True if the user closed the window, else False
            else:
                return False, i.e. continue training.


        """
        self.logger.info('Validating over the entire validation set ({} samples in {} episodes)'.format(
            len(self.validation_problem), len(self.validation_dataloader)))

        # Turn on evaluation mode.
        self.model.eval()

        # Get a random batch index which will be used for visualization
        vis_index = randrange(len(self.validation_dataloader))

        # Reset the statistics.
        self.stat_col.empty()

        with torch.no_grad():
            for ep, valid_batch in enumerate(self.validation_dataloader):
                print(ep)
                # 1. Perform forward step, get predictions and compute loss.
                valid_logits, _ = self.forward_step(self.model, self.validation_problem, valid_batch, ep, epoch)

                # 2.Visualization of validation for the randomly selected batch
                if self.app_state.visualize and ep == vis_index:

                    # Allow for preprocessing
                    valid_batch, valid_logits = self.validation_problem.plot_preprocessing(valid_batch, valid_logits)

                    # Show plot, if user will press Stop then a SystemExit exception will be thrown.
                    self.model.plot(valid_batch, valid_logits)

        # 3. Aggregate statistics.
        self.model.aggregate_statistics(self.stat_col, self.stat_agg)
        self.problem.aggregate_statistics(self.stat_col, self.stat_agg)
        # Set episode, so "the point" will appear in the right place in TB.
        self.stat_agg["episode"] = episode

        # 4. Log to logger
        self.logger.info(self.stat_agg.export_aggregators_to_string('[Validation on the whole set]'))

        # 5. Export to csv
        self.stat_agg.export_aggregators_to_csv(self.validation_stats_aggregated_file)

        if self.validation_writer is not None:
            # Export to TensorBoard.
            self.stat_agg.export_aggregators_to_tensorboard(self.validation_writer)

        # return average loss and whether the user pressed `Quit` during the visualization
        return self.stat_agg['loss']

