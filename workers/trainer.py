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

    - Defines the ``Trainer()`` class, which is the abstract base trainer.


"""
__author__ = "Vincent Marois, Tomasz Kornuta"

import os
import torch
from time import sleep
from random import randrange
from datetime import datetime
from torch.utils.data.dataloader import DataLoader

from workers.worker import Worker
from utils.worker_utils import handshake
from models.model_factory import ModelFactory
from problems.problem_factory import ProblemFactory

from utils.statistics_collector import StatisticsCollector
from utils.statistics_aggregator import StatisticsAggregator


class Trainer(Worker):
    """
    Base class for the trainers.

    Iterates over epochs on the dataset.

    All other types of trainers (e.g. ``ClassicTrainer`` & ``FlexibleTrainer``) should subclass it.

    """

    def __init__(self, name="Trainer"):
        """
        Base constructor for all trainers:

            - Adds default trainer command line arguments

        :param name: Name of the worker (DEFAULT: ''Trainer'').
        :type name: str

        """ 
        # Call base constructor to set up app state, registry and add default params.
        super(Trainer, self).__init__(name)

        # Add arguments to the specific parser.
        # These arguments will be shared by all basic trainers.
        self.parser.add_argument('--tensorboard',
                                 action='store',
                                 dest='tensorboard', choices=[0, 1, 2],
                                 type=int,
                                 help="If present, enable logging to TensorBoard. Available log levels:\n"
                                      "0: Log the collected statistics.\n"
                                      "1: Add the histograms of the model's biases & weights (Warning: Slow).\n"
                                      "2: Add the histograms of the model's biases & weights gradients "
                                      "(Warning: Even slower).")

        self.parser.add_argument('--visualize',
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

    def setup_experiment(self):
        """
        Sets up experiment of all trainers:

            - Calls base class setup_experiment to parse the command line arguments,

            - Loads the config file(s):

                >>> configs_to_load = self.recurrent_config_parse(flags.config, [])

            - Set up the log directory path:

                >>> os.makedirs(self.log_dir, exist_ok=False)

            - Add a ``FileHandler`` to the logger:

                >>>  self.add_file_handler_to_logger(self.log_file)

            - Set random seeds:

                >>>  self.set_random_seeds(self.params['training'], 'training')

            - Creates training problem and model:

                >>> self.training_problem = ProblemFactory.build_problem(self.params['training']['problem'])
                >>> self.model = ModelFactory.build_model(self.params['model'], self.training_problem.default_values)

            - Creates the DataLoader:

                >>> self.training_dataloader = DataLoader(dataset=self.training_problem, ...)

            - Handles curriculum learning if indicated:

                >>> if 'curriculum_learning' in self.params['training']:
                >>> ...

            - Handles the validation of the model:

                - Creates validation problem & DataLoader

            - Set optimizer:

                >>> self.optimizer = getattr(torch.optim, optimizer_name)

            - Handles TensorBoard writers & files:

                >>> self.training_writer = SummaryWriter(self.log_dir + '/training')

        """
        # Call base method to parse all command line arguments and add default sections.
        super(Trainer, self).setup_experiment()

        # Check if config file was selected.
        if self.flags.config == '':
            print('Please pass configuration file(s) as --c parameter')
            exit(-1)

        # Get the list of configurations which need to be loaded.
        configs_to_load = self.recurrent_config_parse(self.flags.config, [])

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
            print("Error: Couldn't retrieve the problem name from the 'training' section in the loaded configuration")
            exit(-1)

        # Get validation problem name
        try:
            _ = self.params['validation']['problem']['name']
        except KeyError:
            print("Error: Couldn't retrieve the problem name from the 'validation' section in the loaded configuration")
            exit(-1)

        # Get model name.
        try:
            model_name = self.params['model']['name']
        except KeyError:
            print("Error: Couldn't retrieve the model name from the loaded configuration")
            exit(-1)

        # Prepare the output path for logging
        while True:  # Dirty fix: if log_dir already exists, wait for 1 second and try again
            try:
                time_str = '{0:%Y%m%d_%H%M%S}'.format(datetime.now())
                if self.flags.savetag != '':
                    time_str = time_str + "_" + self.flags.savetag
                self.log_dir = self.flags.outdir + '/' + training_problem_name + '/' + model_name + '/' + time_str + '/'
                os.makedirs(self.log_dir, exist_ok=False)
            except FileExistsError:
                sleep(1)
            else:
                break

        # Set log dir and add the handler for the logfile to the logger.
        self.log_file = self.log_dir + 'trainer.log'
        self.add_file_handler_to_logger(self.log_file)

        # Models dir.
        self.model_dir = self.log_dir + 'models/'
        os.makedirs(self.model_dir, exist_ok=False)

        # Set random seeds in the training section.
        self.set_random_seeds(self.params['training'], 'training')

        # Check if CUDA is available, if yes turn it on.
        self.check_and_set_cuda(self.params['training'])

        ################# TRAINING PROBLEM ################# 

        # Build the problem for the training
        self.training_problem = ProblemFactory.build_problem(self.params['training']['problem'])

        # build the DataLoader on top of the Problem class, using the associated configuration section.
        self.training_dataloader = DataLoader(dataset=self.training_problem,
                                              batch_size=self.params['training']['problem']['batch_size'],
                                              shuffle=self.params['training']['dataloader']['shuffle'],
                                              sampler=self.params['training']['dataloader']['sampler'],
                                              batch_sampler=self.params['training']['dataloader']['batch_sampler'],
                                              num_workers=self.params['training']['dataloader']['num_workers'],
                                              collate_fn=self.training_problem.collate_fn,
                                              pin_memory=self.params['training']['dataloader']['pin_memory'],
                                              drop_last=self.params['training']['dataloader']['drop_last'],
                                              timeout=self.params['training']['dataloader']['timeout'],
                                              worker_init_fn=self.training_problem.worker_init_fn)

        # parse the curriculum learning section in the loaded configuration.
        if 'curriculum_learning' in self.params['training']:

            # Initialize curriculum learning - with values from loaded configuration.
            self.training_problem.curriculum_learning_initialize(self.params['training']['curriculum_learning'])

            # Set initial values of curriculum learning.
            self.curric_done = self.training_problem.curriculum_learning_update_params(0)

            # If the 'must_finish' key is not present in config then then it will be finished by default
            self.params['training']['curriculum_learning'].add_default_params({'must_finish': True})

            self.must_finish_curriculum = self.params['training']['curriculum_learning']['must_finish']
            self.logger.info("Curriculum Learning activated")

        else:
            # If not using curriculum learning then it does not have to be finished.
            self.must_finish_curriculum = False
            self.curric_done = True

        ################# VALIDATION PROBLEM ################# 
        
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

        # Generate a single batch used for partial validation.
        self.validation_batch = next(iter(self.validation_dataloader))

        ################# MODEL PROBLEM ################# 
        
        # Build the model using the loaded configuration and the default values of the problem.
        self.model = ModelFactory.build_model(self.params['model'], self.training_problem.default_values)

        # load the indicated pretrained model checkpoint if the argument is valid
        if self.flags.model != "":
            if os.path.isfile(self.flags.model):
                # Load parameters from checkpoint.
                self.model.load(self.flags.model)
            else:
                self.logger.error("Couldn't load the checkpoint {} : does not exist on disk.".format(self.flags.model))

        # Move the model to CUDA if applicable.
        if self.app_state.use_CUDA:
            self.model.cuda()

        # Log the model summary.
        self.logger.info(self.model.summarize())

        # perform 2-way handshake between Model and Training Problem
        handshake(model=self.model, problem=self.training_problem, logger=self.logger)
        # no error thrown, so handshake succeeded

        # perform 2-way handshake between Model and Validation Problem
        handshake(model=self.model, problem=self.validation_problem, logger=self.logger)
        # no error thrown, so handshake succeeded

        ################# OPTIMIZER ################# 

        # Set the optimizer.
        optimizer_conf = dict(self.params['training']['optimizer'])
        optimizer_name = optimizer_conf['name']
        del optimizer_conf['name']

        # Instantiate the optimizer and filter the model parameters based on if they require gradients.
        self.optimizer = getattr(torch.optim, optimizer_name)(filter(lambda p: p.requires_grad,
                                                                     self.model.parameters()),
                                                              **optimizer_conf)

    def add_statistics(self, stat_col):
        """
        Calls base method and adds epoch statistics to ``StatisticsCollector``.

        :param stat_col: ``StatisticsCollector``.

        """
        # Add loss and episode.
        super(Trainer, self).add_statistics(stat_col)

        # Add default statistics with formatting.
        stat_col.add_statistic('epoch', '{:02d}')

    def add_aggregators(self, stat_agg):
        """
        Adds basic aggregators to to ``StatisticsAggregator`` and extends them with: epoch.

        :param stat_agg: ``StatisticsAggregator``.

        """
        # Add basic aggregators.
        super(Trainer, self).add_aggregators(stat_agg)

        # add 'aggregators' for the epoch.
        stat_agg.add_aggregator('epoch', '{:02d}')

    def initialize_statistics_collection(self):
        """
        Function initializes all statistics collectors and aggregators used by a given worker, \
        creates output files etc.
        """
        # TRAINING.
        # Create statistics collector for training.
        self.training_stat_col = StatisticsCollector()
        self.add_statistics(self.training_stat_col)
        self.training_problem.add_statistics(self.training_stat_col)
        self.model.add_statistics(self.training_stat_col)
        # Create the csv file to store the training statistics.
        self.training_batch_stats_file = self.training_stat_col.initialize_csv_file(self.log_dir, 'training_statistics.csv')

        # Create statistics aggregator for training.
        self.training_stat_agg = StatisticsAggregator()
        self.add_aggregators(self.training_stat_agg)
        self.training_problem.add_aggregators(self.training_stat_agg)
        self.model.add_aggregators(self.training_stat_agg)
        # Create the csv file to store the training statistic aggregations.
        self.training_set_stats_file = self.training_stat_agg.initialize_csv_file(self.log_dir, 'training_set_agg_statistics.csv')

        # VALIDATION.
        # Create statistics collector for validation.
        self.validation_stat_col = StatisticsCollector()
        self.add_statistics(self.validation_stat_col)
        self.validation_problem.add_statistics(self.validation_stat_col)
        self.model.add_statistics(self.validation_stat_col)
        # Create the csv file to store the validation statistics.
        self.validation_batch_stats_file = self.validation_stat_col.initialize_csv_file(self.log_dir, 'validation_statistics.csv')

        # Create statistics aggregator for validation.
        self.validation_stat_agg = StatisticsAggregator()
        self.add_aggregators(self.validation_stat_agg)
        self.validation_problem.add_aggregators(self.validation_stat_agg)
        self.model.add_aggregators(self.validation_stat_agg)
        # Create the csv file to store the validation statistic aggregations.
        self.validation_set_stats_file = self.validation_stat_agg.initialize_csv_file(self.log_dir, 'validation_set_agg_statistics.csv')

    def finalize_statistics_collection(self):
        """
        Finalizes statistics collection, closes all files etc.

        """
        # Close all files.
        self.training_batch_stats_file.close()
        self.training_set_stats_file.close()
        self.validation_batch_stats_file.close()
        self.validation_set_stats_file.close()

    def initialize_tensorboard(self):
        """
        Initializes TensorBoard writers, and log directories.

        """
        # Create TensorBoard outputs - if TensorBoard is supposed to be used.
        if self.flags.tensorboard is not None:
            from tensorboardX import SummaryWriter
            self.training_batch_writer = SummaryWriter(self.log_dir + '/training')
            self.training_stat_col.initialize_tensorboard(self.training_batch_writer)

            self.training_set_writer = SummaryWriter(self.log_dir + '/training_set_agg')
            self.training_stat_agg.initialize_tensorboard(self.training_set_writer)
            
            self.validation_batch_writer = SummaryWriter(self.log_dir + '/validation')
            self.validation_stat_col.initialize_tensorboard(self.validation_batch_writer)

            self.validation_set_writer = SummaryWriter(self.log_dir + '/validation_set_agg')
            self.validation_stat_agg.initialize_tensorboard(self.validation_set_writer)
        else:
            self.training_batch_writer = None
            self.training_set_writer = None
            self.validation_batch_writer = None
            self.validation_set_writer = None

    def finalize_tensorboard(self):
        """ 
        Finalizes operation of TensorBoard writers.
        """
        # Close the TensorBoard writers.
        if self.training_batch_writer is not None:
            self.training_batch_writer.close()
        if self.training_set_writer is not None:
            self.training_set_writer.close()
        if self.validation_batch_writer is not None:
            self.validation_batch_writer.close()
        if self.validation_set_writer is not None:
            self.validation_set_writer.close()

    def validate_on_batch(self, valid_batch, episode, epoch=None):
        """
        Performs a validation of the model using the provided batch.

        Additionally logs results (to files, TensorBoard) and handles visualization.

        :param valid_batch: data batch generated by the problem and used as input to the model.
        :type valid_batch: ``DataDict``

        :param episode: current training episode index.
        :type episode: int

        :param epoch: current epoch index.
        :type epoch: int, optional

        :return: Validation loss.

        """
        # Turn on evaluation mode.
        self.model.eval()

        # Compute the validation loss using the provided data batch.
        with torch.no_grad():
            valid_logits, valid_loss = self.predict_evaluate_collect(self.model, self.validation_problem,
                                                                     valid_batch, self.validation_stat_col,
                                                                     episode, epoch)

        # Export statistics.
        self.export_statistics(self.validation_stat_col, '[Partial Validation]')

        # Visualization of validation.
        if self.app_state.visualize:
            # Allow for preprocessing
            valid_batch, valid_logits = self.validation_problem.plot_preprocessing(valid_batch, valid_logits)

            # Show plot, if user will press Stop then a SystemExit exception will be thrown.
            self.model.plot(valid_batch, valid_logits)

        return valid_loss

    def validate_on_set(self, episode, epoch=None):
        """
        Performs a validation of the model on the whole validation set, using the validation ``DataLoader``.

        Iterates over the entire validation set (through the `DataLoader``), aggregates the collected statistics \
        and logs that to the console, csv and TensorBoard (if set).

        If visualization is activated, this function will select a random batch to visualize.

        :param episode: current training episode index.
        :type episode: int

        :param epoch: current epoch index.
        :type epoch: int, optional

        :return: Average loss over the validation set.


        """
        self.logger.info('Validating over the entire validation set ({} samples in {} episodes)'.format(
            len(self.validation_problem), len(self.validation_dataloader)))

        # Turn on evaluation mode.
        self.model.eval()

        # Get a random batch index which will be used for visualization
        vis_index = randrange(len(self.validation_dataloader))

        # Reset the statistics.
        self.validation_stat_col.empty()

        with torch.no_grad():
            for ep, valid_batch in enumerate(self.validation_dataloader):
                # 1. Perform forward step, get predictions and compute loss.
                valid_logits, _ = self.predict_evaluate_collect(self.model, self.validation_problem, valid_batch,
                                                                self.validation_stat_col, ep, epoch)

                # 2.Visualization of validation for the randomly selected batch
                if self.app_state.visualize and ep == vis_index:

                    # Allow for preprocessing
                    valid_batch, valid_logits = self.validation_problem.plot_preprocessing(valid_batch, valid_logits)

                    # Show plot, if user will press Stop then a SystemExit exception will be thrown.
                    self.model.plot(valid_batch, valid_logits)

        # Export aggregated statistics.
        self.aggregate_and_export_statistics(self.model, self.validation_problem, 
                self.validation_stat_col, self.validation_stat_agg, episode, '[Full Validation]')

        # Return the average validation loss.
        return self.validation_stat_agg['loss']


if __name__ == '__main__':
    print("The trainer.py file contains only the abstract Trainer class. "
          "Use classic_trainer or flexible_trainer for training.")
