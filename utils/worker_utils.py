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
worker_utils.py: Contains helper functions for different workers.

"""
__author__ = "Ryan McAvoy, Tomasz Kornuta, Vincent Marois"

import os
import yaml

import torch
from .app_state import AppState


def forward_step(model, problem, episode, stat_col, data_dict, epoch=None):
    """
    Function performs a single forward step:

        - passes samples through the model,
        - collects loss & others statistics

    :param model: trainable model.
    :type model: ``models.model.Model`` or a subclass

    :param problem: problem generating samples.
    :type problem: ``problems.problem.problem`` or a subclass

    :param episode: current episode index (either in the current epoch or from the start of the training).
    :type episode: int

    :param stat_col: ``StatisticsCollector``.

    :param data_dict: contains the batch of samples to pass to the model.
    :type data_dict: ``DataDict``

    :param epoch: current epoch index.
    :type epoch: int, optional


    :return:

        - logits,
        - loss

    """
    # convert to CUDA
    if AppState().use_CUDA:
        data_dict = data_dict.cuda()

    # Perform forward calculation.
    logits = model(data_dict)

    # Evaluate loss function.
    loss = problem.evaluate_loss(data_dict, logits)

    # Collect "elementary" statistics - episode and loss.
    if epoch is not None:
        stat_col['epoch'] = epoch

    stat_col['episode'] = episode
    stat_col['loss'] = loss

    # Collect other (potential) statistics from problem & model.
    problem.collect_statistics(stat_col, data_dict, logits)
    model.collect_statistics(stat_col, data_dict, logits)

    # Return tuple: logits, loss.
    return logits, loss


def check_and_set_cuda(params, logger):
    """
    Enables CUDA if available and sets the default data types.

    :param params: Parameter Registry containing either the training or test parameters.
    :type params: ``ParamInterface``

    :param logger: logger object
    :type logger: ``logging.Logger``

    """
    turn_on_cuda = False
    try:  # If the 'cuda' key is not present, catch the exception and do nothing
        turn_on_cuda = params['cuda']
    except KeyError:
        logger.warning('CUDA key not present in ParamInterface.')
        pass

    # Determine if CUDA is to be used.
    if torch.cuda.is_available():
        if turn_on_cuda:
            AppState().convert_cuda_types()
            logger.info('Running with CUDA enabled')
    elif turn_on_cuda:
        logger.warning('CUDA is enabled but there is no available device')

    # TODO Add flags to change these
    AppState().set_dtype('float')
    AppState().set_itype('int')


def recurrent_config_parse(configs: str, configs_parsed: list):
    """
    Parses names of configuration files in a recursive manner, i.e. \
    by looking for ``default_config`` sections and trying to load and parse those
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
            configs_parsed = recurrent_config_parse(
                param_dict['default_configs'], configs_parsed)

    # Done, return list of loaded configs.
    return configs_parsed


def cycle(iterable):
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


def validation(model, problem, episode, stat_col, data_valid, FLAGS, logger, validation_file, validation_writer, epoch=None):
    """
    Performs a validation step on the model, using the provided data batch.

    Additionally logs results (to files, tensorboard) and handles visualization.

    :param model: Neural network model (being trained by the worker) going through cross-validated in this function.
    :type model: ``models.model.Model``

    :param problem: Problem used to generate the batch ``data_valid`` used as input to the model.
    :type problem: ``problems.problem.Problem``

    :param episode: current training episode index.
    :type episode: int

    :param stat_col: statistics collector used for logging accuracy etc.
    :type stat_col: ``StatisticsCollector``

    :param data_valid: data batch generated by the problem and used as input to the model.
    :type data_valid: ``DataDict``

    :param FLAGS: Parsed ``ArgumentParser`` flags
    :type FLAGS: ``argparse.Namespace``

    :param logger: current logger utility.
    :type logger: ``logging.Logger``

    :param validation_file: Opened CSV file used by the ``StatisticsCollector``.

    :param validation_writer: ``tensorboardX.SummaryWriter``.

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
    model.eval()

    # Compute the validation loss using the provided data batch.
    with torch.no_grad():
        logits_valid, loss_valid = forward_step(model, problem, episode, stat_col, data_valid, epoch)

    # Log to logger.
    logger.info(stat_col.export_statistics_to_string('[Validation]'))

    # Export to csv.
    stat_col.export_statistics_to_csv(validation_file)

    if FLAGS.tensorboard is not None:
        # Save loss + accuracy to TensorBoard.
        stat_col.export_statistics_to_tensorboard(validation_writer)

    # Visualization of validation.
    if AppState().visualize:
        # Allow for preprocessing
        data_valid, logits_valid = problem.plot_preprocessing(data_valid, logits_valid)

        # True means that we should terminate
        return loss_valid, model.plot(data_valid, logits_valid)

    # Else simply return false, i.e. continue training.
    return loss_valid, False


def handshake(model, problem, logger):
    """
    Performs the 2-way handshake protocol between the Model and the Problem.

    Sends information to the logger accordingly.

    :param model: Model.
    :type model: ``models.model.Model``

    :param problem: Problem.
    :type problem: ``problems.problem.Problem``

    :param logger: logger.
    :type logger: ``logging.Logger``.

    """

    # perform the first step of the handshake: Ensures that the Model can accept as inputs
    # the batches generated by the Problem.
    if not model.handshake_definitions(problem.data_definitions):
        logger.error('Handshake between {m} and {p} failed: batches generated by {p} do not match {m} expected inputs'
                     '.'.format(p=model.name, m=problem.name))
        exit(-1)
    else:
        logger.info('First Handshake between {m} and {p} succeeded: {m} accepts the batches produced by {p}.'
                    ''.format(m=model.name, p=problem.name))

    # handshake succeeded, so we can continue.

    # perform the second step of the handshake: Ensures that the logits produced by the Model
    # and the ground truth labels of the Problem fits the loss function inputs
    if not problem.handshake_definitions(model.data_definitions):
        logger.error('Handshake between {m} and {p} failed: predictions of {m} or ground truth labels of {p} do not '
                     'match the loss function {l}.'.format(p=model.name, m=problem.name, l=problem.loss_function))
        exit(-1)
    else:
        logger.info('Second Handshake between {m} and {p} succeeded: predictions of {m} & ground truth labels of {p} '
                    'accepted by the loss function {l}.'.format(m=model.name, p=problem.name, l=problem.loss_function))

    # handshake succeeded, so we can continue.
