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
__author__ = "Vincent Marois"

import os
import yaml
import torch
import argparse
import numpy as np
from time import sleep
from datetime import datetime
from torch.nn.utils import clip_grad_value_
from torch.utils.data.dataloader import DataLoader


import workers.worker as worker
from workers.worker import Worker
from models.model_factory import ModelFactory
from problems.problem_factory import ProblemFactory

from utils.worker_utils import forward_step, check_and_set_cuda, recurrent_config_parse, handshake, validate_over_set


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
                        choices=[0, 1, 2, 3],
                        type=int,
                        help="Activate dynamic visualization (Warning: will require user interaction):\n"
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

                >>> torch.manual_seed(self.param_interface["training"]["seed_torch"])
                >>> np.random.seed(self.param_interface["training"]["seed_numpy"])

            - Creates problem and model:

                >>> self.dataset = ProblemFactory.build_problem(self.param_interface['training']['problem'])
                >>> self.model = ModelFactory.build_model(self.param_interface['model'], self.dataset.default_values)

            - Creates the DataLoader:

                >>> self.problem = DataLoader(dataset=self.problem, ...)

            - Handles curriculum learning if indicated:

                >>> if 'curriculum_learning' in self.param_interface['training']:
                >>> ...

            - Handles the validation of the model:

                - Instantiates the problem class, with the parameters contained in the `validation` section,
                - Will validate the model at the end of each epoch, over the entire validation set, and log the \
                statistical estimators (minimum / maximum / average / standard deviation... of the loss, accuracy \
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
            self.param_interface.add_config_params_from_yaml(config)
            print('Loaded configuration from file {}'.format(config))

        # -> At this point, the Param Registry contains the configuration loaded (and overwritten) from several files.

        # Get training problem name.
        try:
            training_problem_name = self.param_interface['training']['problem']['name']
        except KeyError:
            print("Error: Couldn't retrieve problem name from the 'training' section in the loaded configuration")
            exit(-1)

        # Get validation problem name
        try:
            _ = self.param_interface['validation']['problem']['name']
        except KeyError:
            print("Error: Couldn't retrieve problem name from the 'validation' section in the loaded configuration")
            exit(-1)

        # Get model name.
        try:
            model_name = self.param_interface['model']['name']
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

        # Create TensorBoard outputs - if TensorBoard is supposed to be used.
        if flags.tensorboard is not None:
            from tensorboardX import SummaryWriter

            self.training_writer = SummaryWriter(self.log_dir + '/training')
            self.validation_writer = SummaryWriter(self.log_dir + '/validation')
        else:
            self.validation_writer = None

        # set random seeds
        self.set_random_seeds()

        # check if CUDA is available, if yes turn it on
        check_and_set_cuda(self.param_interface['training'], self.logger)

        # Build the problem for the training
        self.problem = ProblemFactory.build_problem(self.param_interface['training']['problem'])

        # check that the number of epochs is available in param_interface. If not, put a default of 1.
        if "max_epochs" not in self.param_interface["training"]["terminal_condition"] \
                or self.param_interface["training"]["terminal_condition"]["max_epochs"] == -1:
            max_epochs = 1

            self.param_interface["training"]["terminal_condition"].add_config_params({'max_epochs': max_epochs})

        self.logger.info("Setting the max number of epochs to: {}".format(
            self.param_interface["training"]["terminal_condition"]["max_epochs"]))

        # ge t theepoch size in terms of episodes:
        epoch_size = self.problem.get_epoch_size(self.param_interface["training"]["problem"]["batch_size"])
        self.logger.info('Epoch size in terms of episodes: {}'.format(epoch_size))

        # Build the model using the loaded configuration and the default values of the problem.
        self.model = ModelFactory.build_model(self.param_interface['model'], self.problem.default_values)

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
                                     batch_size=self.param_interface['training']['problem']['batch_size'],
                                     shuffle=self.param_interface['training']['dataloader']['shuffle'],
                                     sampler=self.param_interface['training']['dataloader']['sampler'],
                                     batch_sampler=self.param_interface['training']['dataloader']['batch_sampler'],
                                     num_workers=self.param_interface['training']['dataloader']['num_workers'],
                                     collate_fn=self.problem.collate_fn,
                                     pin_memory=self.param_interface['training']['dataloader']['pin_memory'],
                                     drop_last=self.param_interface['training']['dataloader']['drop_last'],
                                     timeout=self.param_interface['training']['dataloader']['timeout'],
                                     worker_init_fn=self.problem.worker_init_fn)

        # parse the curriculum learning section in the loaded configuration.
        if 'curriculum_learning' in self.param_interface['training']:

            # Initialize curriculum learning - with values from loaded configuration.
            self.problem.curriculum_learning_initialize(self.param_interface['training']['curriculum_learning'])

            # Set initial values of curriculum learning.
            self.curric_done = self.problem.curriculum_learning_update_params(0)

            # If the 'must_finish' key is not present in config then then it will be finished by default
            if 'must_finish' not in self.param_interface['training']['curriculum_learning']:
                self.param_interface['training']['curriculum_learning'].add_default_params({'must_finish': True})

            self.must_finish_curriculum = self.param_interface['training']['curriculum_learning']['must_finish']
            self.logger.info("Using curriculum learning")

        else:
            # Initialize curriculum learning - with empty dict.
            self.problem.curriculum_learning_initialize({})

            # If not using curriculum learning then it does not have to be finished.
            self.must_finish_curriculum = False

        # Add the model & problem dependent statistics to the ``StatisticsCollector``
        self.problem.add_statistics(self.stat_col)
        self.model.add_statistics(self.stat_col)

        # Add the model & problem dependent statistical estimators to the ``StatisticsEstimators``
        self.problem.add_estimators(self.stat_est)
        self.model.add_estimators(self.stat_est)

        # Create the csv file to store the training statistics.
        self.training_stats_file = self.stat_col.initialize_csv_file(self.log_dir, 'training_statistics.csv')

        # Build the validation problem
        self.problem_validation = ProblemFactory.build_problem(self.param_interface['validation']['problem'])

        # build the DataLoader on top of the validation problem
        self.dl_valid = DataLoader(dataset=self.problem_validation,
                                   batch_size=self.param_interface['validation']['problem']['batch_size'],
                                   shuffle=self.param_interface['validation']['dataloader']['shuffle'],
                                   sampler=self.param_interface['validation']['dataloader']['sampler'],
                                   batch_sampler=self.param_interface['validation']['dataloader']['batch_sampler'],
                                   num_workers=self.param_interface['validation']['dataloader']['num_workers'],
                                   collate_fn=self.problem_validation.collate_fn,
                                   pin_memory=self.param_interface['validation']['dataloader']['pin_memory'],
                                   drop_last=self.param_interface['validation']['dataloader']['drop_last'],
                                   timeout=self.param_interface['validation']['dataloader']['timeout'],
                                   worker_init_fn=self.problem_validation.worker_init_fn)

        # Create the csv file to store the validation statistical estimators
        # This file will contains several data points for the ``Trainer`` (but only one for the ``EpisodicTrainer``)
        self.val_est_file = self.stat_est.initialize_csv_file(self.log_dir, 'validation_estimators.csv')

        # Set the optimizer.
        optimizer_conf = dict(self.param_interface['training']['optimizer'])
        optimizer_name = optimizer_conf['name']
        del optimizer_conf['name']

        # Instantiate the optimizer and filter the model parameters based on if they require gradients.
        self.optimizer = getattr(torch.optim, optimizer_name)(filter(lambda p: p.requires_grad,
                                                                     self.model.parameters()),
                                                              **optimizer_conf)

        # -> At this point, all configuration for the ``Trainer`` is complete.

        # -------------------------
        # This attribute is used for the ``EpisodicTrainer``: as this constructor is also used by the
        # ``EpisodicTrainer``, we set it here before saving the configuration to file.
        # Set the Model validation frequency for the ``EpisodicTrainer`` (Default: 100 episodes).
        try:
            self.model_validation_interval = self.param_interface['training']['validation_interval']
        except KeyError:
            self.model_validation_interval = 100
        # -------------------------

        # Save the resulting configuration into a .yaml settings file, under log_dir
        with open(self.log_dir + "training_configuration.yaml", 'w') as yaml_backup_file:
            yaml.dump(self.param_interface.to_dict(), yaml_backup_file, default_flow_style=False)

        # Log the resulting training configuration.
        conf_str = '\n' + '='*80 + '\n'
        conf_str += 'Final registry configuration for training {} on {}:\n'.format(model_name, training_problem_name)
        conf_str += '='*80 + '\n'
        conf_str += yaml.safe_dump(self.param_interface.to_dict(), default_flow_style=False)
        conf_str += '='*80 + '\n'
        self.logger.info(conf_str)

    def forward(self, flags: argparse.Namespace):
        """
        Main function of the ``Trainer``.

        Iterates over the number of epochs and the ``DataLoader`` (representing the training set).

        .. note::

            Because of the export of the weights and gradients to TensorBoard, we need to\
             keep track of the current episode index from the start of the training, even \
            though the Worker runs on epoch. This may change in a future release.

        .. warning::

            The test for terminal conditions (e.g. convergence) is done at the end of each epoch, \
            not episode. The terminal conditions are as follows:

                - The loss is below the specified threshold (using the validation loss),
                - Early stopping is set and the validation loss did not change by delta for the indicated number \
                of epochs, (todo: coming in a future release)
                - The maximum number of epochs has been met.


        The function does the following for each epoch:

            - Executes the ``initialize_epoch()`` & ``finish_epoch()`` function of the ``Problem`` class,
            - Iterates over the ``DataLoader``, and for each episode:

                    - Handles curriculum learning if set,
                    - Resets the gradients
                    - Forwards pass of the model,
                    - Logs statistics and exports to tensorboard (if set),
                    - Computes gradients and update weights
                    - Activate visualization if set (vis. level 0)

            - Validate the model on the entire validation set, logs the statistical estimators values \
              and visualize on a randon batch if set (vis. level 1 or 2)


        A last validation on the entire set is done at the end on training (if a terminal condition is met), \
        and visualize a random batch if set (vis. level 3).



        """
        # Create the csv file to store the training statistical estimators.
        # doing it in the forward, not constructor, as the ``EpisodicTrainer`` does not need it.
        self.training_est_file = self.stat_est.initialize_csv_file(self.log_dir, 'training_estimators.csv')

        # Ask for confirmation - optional.
        if flags.confirm:
            input('Press any key to continue')

        # Flag denoting whether we converged.
        converged = False

        '''
        Main training and validation loop.
        '''
        episode = 0

        for epoch in range(self.param_interface["training"]["terminal_condition"]["max_epochs"]):

            # user_pressed_stop = True means that we stop visualizing training episodes for the current epoch.
            user_pressed_stop = False

            # empty Statistics Collector
            # note: The StatisticsCollector is emptied once per epoch. If the epoch size is large, this may cause a
            # high memory usage. todo: how to systematically prevent this?
            self.stat_col.empty()
            self.logger.info('Emptied StatisticsCollector.')

            self.logger.info('Epoch {} started'.format(epoch))
            # tell the problem class that epoch has started: can be used to set / reset counters etc.
            self.problem.initialize_epoch(epoch)

            # iterate over training set
            for data_dict in self.dataloader:

                # reset all gradients
                self.optimizer.zero_grad()

                # Set visualization flag if visualization is wanted during training & validation episodes.
                if flags.visualize is not None and flags.visualize <= 1:
                    self.app_state.visualize = True
                else:
                    self.app_state.visualize = False

                # Turn on training mode for the model.
                self.model.train()

                # 1. Perform forward step, get predictions, compute loss and log statistics.
                logits, loss = forward_step(self.model, self.problem, episode, self.stat_col, data_dict, epoch)

                # 2. Backward gradient flow.
                loss.backward()

                # Check the presence of the 'gradient_clipping'  parameter.
                try:
                    # if present - clip gradients to a range (-gradient_clipping, gradient_clipping)
                    val = self.param_interface['training']['gradient_clipping']
                    clip_grad_value_(self.model.parameters(), val)

                except KeyError:
                    # Else - do nothing.
                    pass

                # 3. Perform optimization.
                self.optimizer.step()

                # 4. Log collected statistics.

                # 4.1. Log to logger according to the logging frequency.
                if episode % flags.logging_frequency == 0:
                    self.logger.info(self.stat_col.export_statistics_to_string())

                # 4.2. Export to csv.
                self.stat_col.export_statistics_to_csv(self.training_stats_file)

                # 4.3. Export data to tensorboard according to the logging frequency.
                if (flags.tensorboard is not None) and (episode % flags.logging_frequency == 0):
                    self.stat_col.export_statistics_to_tensorboard(self.training_writer)

                    # Export histograms.
                    if flags.tensorboard >= 1:
                        for name, param in self.model.named_parameters():
                            try:
                                self.training_writer.add_histogram(name, param.data.cpu().numpy(), episode, bins='doane')

                            except Exception as e:
                                self.logger.error("  {} :: data :: {}".format(name, e))

                    # Export gradients.
                    if flags.tensorboard >= 2:
                        for name, param in self.model.named_parameters():
                            try:
                                self.training_writer.add_histogram(name + '/grad', param.grad.data.cpu().numpy(),
                                                                   episode, bins='doane')

                            except Exception as e:
                                self.logger.error("  {} :: grad :: {}".format(name, e))

                # 5. Authorize visualization if the flag is set (vis level 0 or 1) and if the user did not previously
                # clicked 'Stop visualization' during the current epoch.
                if self.app_state.visualize and not user_pressed_stop:

                    # Allow for preprocessing
                    data_dict, logits = self.problem.plot_preprocessing(data_dict, logits)
                    # visualization
                    user_pressed_stop = self.model.plot(data_dict, logits)

                episode += 1

            # Finalize the epoch
            self.logger.info('Epoch {} finished'.format(epoch))

            # Collect the statistical estimators
            self.model.collect_estimators(self.stat_col, self.stat_est)
            self.problem.collect_estimators(self.stat_col, self.stat_est)

            self.stat_est['episode'] = episode
            self.stat_est['epoch'] = epoch

            # Log the statistical estimators to the logger
            self.logger.info(self.stat_est.export_estimators_to_string())
            # Log the statistical estimators to the csv file
            self.stat_est.export_estimators_to_csv(self.training_est_file)

            # empty Statistics Collector
            self.stat_col.empty()
            self.logger.info('Emptied StatisticsCollector.')

            # tell the problem class that the epoch has ended
            self.problem.finalize_epoch(epoch)

            # 5. Validate over the entire validation set
            # Check visualization flag
            if flags.visualize is not None and (1 <= flags.visualize <= 2):
                self.app_state.visualize = True
            else:
                self.app_state.visualize = False
            avg_loss_valid, _ = validate_over_set(self.model, self.problem_validation, self.dl_valid,
                                                  self.stat_col, self.stat_est, flags, self.logger,
                                                  self.val_est_file, self.validation_writer, epoch)

            # Save the model using the average validation loss.
            self.model.save(self.model_dir, avg_loss_valid, self.stat_est)

            # 6. Terminal conditions: Tests which conditions have been met.

            # apply curriculum learning - change some of the Problem parameters
            self.curric_done = self.problem.curriculum_learning_update_params(episode)


            # I. the loss is < threshold (only when curriculum learning is finished if set.)
            if self.curric_done or not self.must_finish_curriculum:

                # loss_stop = True if convergence
                loss_stop = self.stat_est['loss_mean'] < self.param_interface['training']['terminal_condition']['loss_stop']

                if loss_stop:
                    # Ok, we have converged.
                    converged = True
                    terminal_condition = 'Loss < Threshold.'

                    # Finish the training.
                    break

            # II. Early stopping is set and loss hasn't improved by delta in n epochs.
            # early_stopping(index=epoch, avg_loss_valid). (todo: coming in next release)
            # converged = False
            # terminal_condition = 'Early Stopping.'

            # III - The epochs number limit has been reached.
            if epoch > self.param_interface['training']['terminal_condition']['max_epochs']:
                terminal_condition = 'Maximum number of epochs reached.'
                # If we reach this condition, then it is possible that the model didn't converge correctly
                # and presents poorer performance.
                converged = False

                # We still save the model as it may perform better during this epoch
                # (as opposed to the previous checkpoint)

                # Validate on the problem if required - so we can collect the
                # statistics needed during saving of the best model.
                _, _ = validation(self.model, self.problem_validation, episode,
                            self.stat_col, self.data_valid, flags, self.logger,
                            self.validation_file, self.validation_writer, epoch)
                # save the model
                self.model.save(self.model_dir, self.stat_col)
                # "Finish" the training.
                break

        '''
        End of main training and validation loop.
        '''
        # log which terminal condition has been met.
        self.logger.info('Learning finished: Met the following terminal condition: {}'.format(terminal_condition))

        # indicate whether the model has converged or not.
        self.logger.info('Converged = {}'.format(converged))

        # Check visualization flag - turn on visualization for last validation if needed.
        if flags.visualize is not None and (flags.visualize == 3):
            self.app_state.visualize = True
        else:
            self.app_state.visualize = False

        # Perform last validation (mainly for if flags.visualize = 3 since we just validated this model).
        self.logger.info('Last validation on the entire validation set:')
        _, _ = validate_over_set(self.model, self.problem_validation, self.dl_valid, self.stat_col,
                                 self.stat_est, flags, self.logger, self.val_est_file, self.validation_writer,
                                 'after end of training.')

        # Close all files.
        self.training_stats_file.close()
        self.training_est_file.close()
        self.val_est_file.close()

        if flags.tensorboard is not None:
            # Close the TensorBoard writers.
            self.training_writer.close()
            self.validation_writer.close()


if __name__ == '__main__':
    # Create parser with list of  runtime arguments.
    argp = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # add default arguments
    worker.add_arguments(argp)

    # add trainers-specific arguments
    add_arguments(argp)

    # Parse arguments.
    FLAGS, unparsed = argp.parse_known_args()

    trainer = Trainer(FLAGS)
    trainer.forward(FLAGS)
