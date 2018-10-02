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
base_trainer.py:

    - This file sets hosts a function which adds specific arguments a trainer will need.
    - Also defines the BaseTrainer() class.


"""
__author__ = "Vincent Marois"

import os
import yaml
import torch
import logging
import argparse
import collections
import numpy as np
from time import sleep
from random import randrange
from datetime import datetime
from torch.nn.utils import clip_grad_value_
from torch.utils.data.dataloader import DataLoader


import workers.base_worker as worker
from workers.base_worker import BaseWorker
from models.model_factory import ModelFactory
from problems.problem_factory import ProblemFactory

from utils.worker_utils import forward_step, check_and_set_cuda, recurrent_config_parse, validation, handshake


def add_arguments(parser: argparse.ArgumentParser):
    """
    Add arguments to the specific parser.
    These arguments will be shared by all (basic) trainers.
    :param parser: ``argparse.ArgumentParser``
    """
    # add here all arguments used by the trainers.
    parser.add_argument('--agree',
                        dest='confirm',
                        action='store_true',
                        help='Request user confirmation just after loading the settings, '
                             'before starting training  (Default: False)')

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
                        help='Path to output directory where the experiment(s) folders will be stored.'
                             ' (DEFAULT: ./experiments)')

    parser.add_argument('--tensorboard',
                        action='store',
                        dest='tensorboard', choices=[0, 1, 2],
                        type=int,
                        help="If present, enable log to TensorBoard. Available log levels:\n"
                             "0: Log loss, accuracy, and collected statistics.\n"
                             "1: Add the histograms of the model's biases & weights (Warning: Slow).\n"
                             "2: Add the histograms of the model's biases & weights gradients (Warning: Even slower).")

    parser.add_argument('--lf',
                        dest='logging_frequency',
                        default=100,
                        type=int,
                        help='TensorBoard logging frequency (Default: 100, i.e. logs every 100 episodes).')

    parser.add_argument('--visualize',
                        dest='visualize',
                        choices=[0, 1, 2, 3],
                        type=int,
                        help="Activate dynamic visualization (Warning: will require user interaction):\n"
                             "0: Only during training episodes.\n"
                             "1: During both training and validation episodes.\n"
                             "2: Only during validation episodes.\n"
                             "3: Only during the last validation episode, after training is completed.\n")


class BaseTrainer(BaseWorker):
    """
    Base class for the trainers.

    All trainers should subclass it.

    """

    def __init__(self, flags: argparse.Namespace):
        """
        Base constructor for all trainers:

            - Loads the config file(s)
            - Set up the log directory path
            - Add a FileHandler to the logger (defined in BaseWorker)
            - Handles TensorBoard writers & files
            - Set random seeds
            - Creates problem and model
            - Handles curriculum learning if indicated
            - Set optimizer


        :param flags: Parsed arguments from the parser.

        TODO: Enhance documentation
        """
        # default name (define it before calling base constructor for logger)
        self.name = 'BaseTrainer'

        # call base constructor
        super(BaseTrainer, self).__init__(flags)

        # Check if config file was selected.
        if flags.config == '':
            print('Please pass configuration file(s) as --c parameter')
            exit(-1)

        # Get list of configs that need to be loaded.
        configs_to_load = recurrent_config_parse(flags.config, [])

        # Read the YAML files one by one - but in reverse order!
        for config in reversed(configs_to_load):
            # Open file and try to add that to list of parameter dictionaries.
            with open(config, 'r') as stream:
                # Load param dictionaries in reverse order.
                self.param_interface.add_custom_params(yaml.load(stream))

            print('Loaded configuration from file {}'.format(config))
            # Add to list of loaded configs.

            configs_to_load.append(config)

        # -> At this point, the Param Registry contains the configuration loaded (and overwritten) from several files.

        # Get problem name.
        try:
            task_name = self.param_interface['training']['problem']['name']
        except BaseException:
            print("Error: Couldn't retrieve problem name from the loaded configuration")
            exit(-1)

        # Get model name.
        try:
            model_name = self.param_interface['model']['name']
        except BaseException:
            print("Error: Couldn't retrieve model name from the loaded configuration")
            exit(-1)

        # Prepare output paths for logging
        while True:  # Dirty fix: if log_dir already exists, wait for 1 second and try again
            try:
                time_str = '{0:%Y%m%d_%H%M%S}'.format(datetime.now())
                if flags.savetag != '':
                    time_str = time_str + "_" + flags.savetag
                self.log_dir = flags.outdir + '/' + task_name + '/' + model_name + '/' + time_str + '/'
                os.makedirs(self.log_dir, exist_ok=False)
            except FileExistsError:
                sleep(1)
            else:
                break

        self.model_dir = self.log_dir + 'models/'
        os.makedirs(self.model_dir, exist_ok=False)
        self.log_file = self.log_dir + 'trainer.log'

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

        # Create tensorboard output - if tensorboard is supposed to be used.
        if flags.tensorboard is not None:
            from tensorboardX import SummaryWriter

            self.training_writer = SummaryWriter(self.log_dir + '/training')
            self.validation_writer = SummaryWriter(self.log_dir + '/validation')
        else:
            self.validation_writer = None

        # Set the random seeds: either from the loaded configuration or a default randomly selected one.
        if "seed_torch" not in self.param_interface["training"] or self.param_interface["training"]["seed_torch"] == -1:
            seed = randrange(0, 2 ** 32)
            self.param_interface["training"].add_custom_params({"seed_torch": seed})
        self.logger.info("Setting torch random seed to: {}".format(self.param_interface["training"]["seed_torch"]))
        torch.manual_seed(self.param_interface["training"]["seed_torch"])
        torch.cuda.manual_seed_all(self.param_interface["training"]["seed_torch"])

        if "seed_numpy" not in self.param_interface["training"] or self.param_interface["training"]["seed_numpy"] == -1:
            seed = randrange(0, 2 ** 32)
            self.param_interface["training"].add_custom_params({"seed_numpy": seed})
        self.logger.info("Setting numpy random seed to: {}".format(self.param_interface["training"]["seed_numpy"]))
        np.random.seed(self.param_interface["training"]["seed_numpy"])

        # check if CUDA is available turn it on
        check_and_set_cuda(self.param_interface['training'], self.logger)

        # Build problem for the training
        self.dataset = ProblemFactory.build_problem(self.param_interface['training']['problem'])

        # Build the model using the loaded configuration and default values of the problem.
        self.model = ModelFactory.build_model(self.param_interface['model'], self.dataset.default_values)

        # move model to CUDA if applicable
        self.model.cuda() if self.app_state.use_CUDA else None

        # perform 2-way handshake between Model and Problem
        handshake(model=self.model, problem=self.dataset, logger=self.logger)
        # no error thrown, so handshake succeeded

        # build the DataLoader on top of the Problem class
        # Set a default number of workers to 4
        # TODO: allow the user to change the num_workers and other attributes value of the DataLoader.
        self.problem = DataLoader(dataset=self.dataset,
                                  batch_size=self.param_interface['training']['problem']['batch_size'],
                                  shuffle=True,
                                  collate_fn=self.dataset.collate_fn,
                                  num_workers=4,
                                  worker_init_fn=self.dataset.worker_init_fn)

        # parse the curriculum learning section in the loaded configuration.
        if 'curriculum_learning' in self.param_interface['training']:

            # Initialize curriculum learning - with values from loaded configuration.
            self.dataset.curriculum_learning_initialize(self.param_interface['training']['curriculum_learning'])

            # Set initial values of curriculum learning.
            self.curric_done = self.dataset.curriculum_learning_update_params(0)

            # If key is not present in config then it has to be finished (DEFAULT: True)
            if 'must_finish' not in self.param_interface['training']['curriculum_learning']:
                self.param_interface['training']['curriculum_learning'].add_default_params({'must_finish': True})

            self.must_finish_curriculum = self.param_interface['training']['curriculum_learning']['must_finish']
            self.logger.info("Using curriculum learning")

        else:
            # Initialize curriculum learning - with empty dict.
            self.dataset.curriculum_learning_initialize({})

            # If not using curriculum learning then it does not have to be finished.
            self.must_finish_curriculum = False

        # Set the Model validation frequency (DEFAULT: 100 episodes).
        try:
            self.model_validation_interval = self.param_interface['training']['validation_interval']
        except KeyError:
            self.model_validation_interval = 100

        # Add model/problem dependent statistics.
        self.dataset.add_statistics(self.stat_col)
        self.model.add_statistics(self.stat_col)

        # Create the csv file to store the training statistics.
        self.training_file = self.stat_col.initialize_csv_file(self.log_dir, 'training.csv')

        # Check if the validation section is present AND problem section is also present...
        if ('validation' in self.param_interface) and ('problem' in self.param_interface['validation']):
            # ... then load problem, set variables etc.

            # Build the validation problem
            self.problem_validation = ProblemFactory.build_problem(self.param_interface['validation']['problem'])

            # build the DataLoader on top of the validation problem
            # Set a default number of workers to 4
            # For now, it doesn't use a Sampler: only shuffling the data.
            dataloader_validation = DataLoader(self.problem_validation,
                                               batch_size=self.param_interface['validation']['problem']['batch_size'],
                                               shuffle=True,
                                               collate_fn=self.problem_validation.collate_fn,
                                               num_workers=4,
                                               worker_init_fn=self.problem_validation.worker_init_fn)
            # create an iterator
            dataloader_validation = iter(dataloader_validation)

            # Get a single batch that will be used for validation (!)
            # TODO: move this step in validation() and handle using more than 1 batch for validation.
            self.data_valid = next(dataloader_validation)

            # Create the csv file to store the validation statistics.
            self.validation_file = self.stat_col.initialize_csv_file(self.log_dir, 'validation.csv')

            # Turn on validation.
            self.use_validation_problem = True
            self.logger.info("Using validation problem for calculation of loss and model validation")

        else:
            # We do not have a validation problem - so turn it off.
            self.use_validation_problem = False
            self.logger.info("Using training problem for calculation of loss and model validation")

            # Use the training loss instead as a convergence criterion: If the loss is < threshold for a given\
            # number of episodes (default: 10), assume the model has converged.
            try:
                self.loss_length = self.param_interface['training']['length_loss']
            except KeyError:
                self.loss_length = 10

        # Set the optimizer.
        optimizer_conf = dict(self.param_interface['training']['optimizer'])
        optimizer_name = optimizer_conf['name']
        del optimizer_conf['name']

        # Select for optimization only the parameters that require update!
        self.optimizer = getattr(torch.optim, optimizer_name)(filter(lambda p: p.requires_grad,
                                                                     self.model.parameters()),
                                                              **optimizer_conf)

        # -> At this point, all configuration is complete.
        # Save the resulting configuration into a .yaml settings file, under log_dir
        with open(self.log_dir + "training_configuration.yaml", 'w') as yaml_backup_file:
            yaml.dump(self.param_interface.to_dict(), yaml_backup_file, default_flow_style=False)

        # Print the training configuration.
        infostr = 'Configuration for {}:\n'.format(task_name)
        infostr += yaml.safe_dump(self.param_interface.to_dict(), default_flow_style=False,
                                  explicit_start=True, explicit_end=True)
        self.logger.info(infostr)

    def forward(self, flags: argparse.Namespace):
        """
        TODO: Make documentation
        """

        # check that the number of epochs is available in param_interface. If not, put a default of 1.
        if "max_epochs" not in self.param_interface["training"]["terminal_condition"] \
                or self.param_interface["training"]["terminal_condition"]["max_epochs"] == -1:
            max_epochs = 1

            self.param_interface["training"]["terminal_condition"].add_custom_params({'max_epochs': max_epochs})

        self.logger.info("Setting the max number of epochs to: {}".format(
            self.param_interface["training"]["terminal_condition"]["max_epochs"]))

        # get epoch size in terms of episodes:
        epoch_size = self.dataset.get_epoch_size(self.param_interface["training"]["problem"]["batch_size"])
        self.logger.info('Epoch size in terms of episodes: {}'.format(epoch_size))

        # Ask for confirmation - optional.
        if flags.confirm:
            input('Press any key to continue')

        # Start Training
        last_losses = collections.deque()

        # Flag denoting whether we converged (or reached last episode).
        terminal_condition = False

        '''
        Main training and validation loop.
        '''
        user_pressed_stop = False
        episode = 0

        for epoch in range(self.param_interface["training"]["terminal_condition"]["max_epochs"]):

            # initialize the epoch: this function can be used to set / reset counters etc.
            self.logger.info('')
            self.logger.info('Epoch {} started'.format(epoch))
            self.logger.info('')
            self.dataset.initialize_epoch(epoch)

            validation_loss = np.inf

            for data_dict in self.problem:

                # apply curriculum learning - change some of the Problem parameters
                self.curric_done = self.dataset.curriculum_learning_update_params(episode)

                # reset all gradients
                self.optimizer.zero_grad()

                # Set visualization flag if visualization is wanted during training & validation episodes.
                if flags.visualize is not None and flags.visualize <= 1:
                    self.app_state.visualize = True
                else:
                    self.app_state.visualize = False

                # Turn on training mode for the model.
                self.model.train()

                # 1. Perform forward step, get predictions and compute loss.
                logits, loss = forward_step(self.model, self.dataset, episode, self.stat_col, data_dict, epoch)

                if not self.use_validation_problem:

                    # Store the calculated loss on a list.
                    last_losses.append(loss)

                    # Truncate list length.
                    if len(last_losses) > self.loss_length:
                        last_losses.popleft()

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

                # 4. Log statistics.

                # Log to logger.
                self.logger.info(self.stat_col.export_statistics_to_string())

                # Export to csv.
                self.stat_col.export_statistics_to_csv(self.training_file)

                # Export data to tensorboard.
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

                # Check visualization of training data.
                if self.app_state.visualize:

                    # Allow for preprocessing
                    data_dict, logits = self.dataset.plot_preprocessing(data_dict, logits)

                    # Show plot, if user presses Quit - break.
                    if self.model.plot(data_dict, logits):
                        user_pressed_stop = True
                        break

                #  5. Validate and (optionally) save the model.

                if (episode % self.model_validation_interval) == 0:

                    # Validate on the problem if required.
                    if self.use_validation_problem:

                        # Check visualization flag
                        if flags.visualize is not None and (flags.visualize == 1 or flags.visualize == 2):
                            self.app_state.visualize = True
                        else:
                            self.app_state.visualize = False

                        # Perform validation.
                        validation_loss, user_pressed_stop = validation(self.model, self.problem_validation, episode,
                                                                        self.stat_col, self.data_valid, flags, self.logger,
                                                                        self.validation_file, self.validation_writer, epoch)

                    # Save the model using the latest (validation or training) statistics.
                    self.model.save(self.model_dir, self.stat_col)

                episode += 1

            # finalize the epoch, even if the learning was interrupted
            self.logger.info('')
            self.logger.info('Epoch {} finished'.format(epoch))
            self.logger.info('')
            self.dataset.finalize_epoch(epoch)

            # 6. Terminal conditions.

            # I. The User pressed stop during visualization.
            if user_pressed_stop:
                terminal_condition = False
                break

            # II. & III - the loss is < threshold (only when curriculum learning is finished if set.)
            if self.curric_done or not self.must_finish_curriculum:

                # break if convergence
                if self.use_validation_problem:
                    loss_stop = validation_loss < self.param_interface['training']['terminal_condition']['loss_stop']
                    # We already saved that model.
                else:
                    loss_stop = max(last_losses) < self.param_interface['training']['terminal_condition']['loss_stop']
                    # We already saved that model.

                if loss_stop:
                    # Ok, we have converged.
                    terminal_condition = True
                    # "Finish" the training.
                    break

            # IV - The epochs number limit has been reached.
            if epoch == self.param_interface['training']['terminal_condition']['max_epochs']:
                terminal_condition = True
                # If we reach this condition, then it is possible that the model didn't converge correctly
                # and present poorer performance.

                # We still save the model as it may perform better during this epoch
                # (as opposed to the previous episode)

                # Validate on the problem if required - so we can collect the
                # statistics needed during saving of the best model.
                if self.use_validation_problem:
                    validation_loss, user_pressed_stop = validation(self.model, self.problem_validation, episode,
                                                                    self.stat_col, self.data_valid, flags, self.logger,
                                                                    self.validation_file, self.validation_writer, epoch)
                # save the model
                self.model.save(self.model_dir, self.stat_col)
                # "Finish" the training.
                break

        '''
        End of main training and validation loop.
        '''

        # Check whether we have finished training properly.
        if terminal_condition:

            self.logger.info('Learning finished!')
            # Check visualization flag - turn on visualization for last validation if needed.
            if flags.visualize is not None and (flags.visualize == 3):
                self.app_state.visualize = True

                # Perform validation.
                if self.use_validation_problem:
                    _, _ = validation(self.model, self.problem_validation, episode, self.stat_col, self.data_valid,
                                      flags, self.logger, self.validation_file, self.validation_writer)

            else:
                self.app_state.visualize = False

        else:  # the training did not end properly
            self.logger.warning('Learning interrupted!')

        # Close all files.
        self.training_file.close()
        self.validation_file.close()

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

    base_trainer = BaseTrainer(FLAGS)
    base_trainer.forward(FLAGS)
