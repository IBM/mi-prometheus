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

"""trainer.py: contains code of worker realising training using CPUs/GPUs"""
__author__      = "Alexis Asseman, Ryan McAvoy, Tomasz Kornuta, Vincent Albouy"


import logging
import logging.config
import os

# Force MKL (CPU BLAS) to use one core, faster
os.environ["OMP_NUM_THREADS"] = '1'

import yaml
from random import randrange

from datetime import datetime
from time import sleep

import argparse
import torch
from torch import nn
import collections
import numpy as np

# Import utils.
from utils.app_state import AppState
from utils.statistics_collector import StatisticsCollector
from utils.param_interface import ParamInterface
from utils.worker_utils import forward_step, check_and_set_cuda, recurrent_config_parse

# Import model and problem factories.
from problems.problem_factory import ProblemFactory
from models.model_factory import ModelFactory

def validation(
        model,
        problem,
        episode,
        stat_col,
        data_valid,
        aux_valid,
        FLAGS,
        logger,
        validation_file,
        validation_writer):
    """
    Function performs validation of the model, using the provided data and
    criterion. Additionally it logs (to files, tensorboard) and visualizes.

    :param stat_col: Statistic collector object.
    :return: True if training loop is supposed to end.

    """
    # Turn on evaluation mode.
    model.eval()
    # Calculate loss of the validation data.
    with torch.no_grad():
        logits_valid, loss_valid = forward_step(
            model, problem, episode, stat_col, data_valid, aux_valid)

    # Log to logger.
    logger.info(stat_col.export_statistics_to_string('[Validation]'))
    # Export to csv.
    stat_col.export_statistics_to_csv(validation_file)

    if (FLAGS.tensorboard is not None):
        # Save loss + accuracy to tensorboard.
        stat_col.export_statistics_to_tensorboard(validation_writer)

    # Visualization of validation.
    if AppState().visualize:
        # True means that we should terminate

        # Allow for preprocessing
        data_valid, aux_valid, logits_valid = problem.plot_preprocessing(
            data_valid, aux_valid, logits_valid)

        return loss_valid, model.plot(data_valid, logits_valid)
    # Else simply return false, i.e. continue training.
    return loss_valid, False


if __name__ == '__main__':
    # Create parser with list of  runtime arguments.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--agree',
        dest='confirm',
        action='store_true',
        help='Request user confirmation just after loading the settings, before starting training  (Default: False)')
    parser.add_argument(
        '--config',
        dest='config',
        type=str,
        default='',
        help='Name of the configuration file(s) to be loaded (more than one file must be separated with coma ",")')
    parser.add_argument('--savetag', dest='savetag', type=str, default='',
                        help='Tag for the save directory')
    parser.add_argument(
        '--outdir',
        dest='outdir',
        type=str,
        default="./experiments",
        help='Path to output directory where the experiments will be stored (DEFAULT: ./experiments)12')
    parser.add_argument(
        '--tensorboard', action='store', dest='tensorboard', choices=[0, 1, 2],
        type=int,
        help="If present, log to TensorBoard. Log levels:\n"
        "0: Just log the loss, accuracy, and seq_len\n"
        "1: Add histograms of biases and weights (Warning: slow)\n"
        "2: Add histograms of biases and weights gradients (Warning: even slower)")
    parser.add_argument(
        '--lf', dest='logging_frequency', default=100, type=int,
        help='TensorBoard logging frequency (Default: 100, i.e. logs every 100 episodes)')
    parser.add_argument(
        '--log',
        action='store',
        dest='log',
        type=str,
        default='INFO',
        choices=[
            'CRITICAL',
            'ERROR',
            'WARNING',
            'INFO',
            'DEBUG',
            'NOTSET'],
        help="Log level. (Default: INFO)")
    parser.add_argument(
        '--visualize',
        dest='visualize',
        choices=[
            0,
            1,
            2,
            3],
        type=int,
        help="Activate dynamic visualization:\n"
        "0: Only during training\n"
        "1: During both training and validation\n"
        "2: Only during validation\n"
        "3: Only during last validation, after training is completed\n")

    # Parse arguments.
    FLAGS, unparsed = parser.parse_known_args()

    # Check if config file was selected.
    if FLAGS.config == '':
        print('Please pass configuration file(s) as --c parameter')
        exit(-1)

    # Get list of configs that need to be loaded.
    configs_to_load = recurrent_config_parse(FLAGS.config, [])

    # Create param interface object.
    param_interface = ParamInterface()

    # Read the YAML files one by one - but in reverse order!
    for config in reversed(configs_to_load):
        # Open file and try to add that to list of parameter dictionaries.
        with open(config, 'r') as stream:
            # Load param dictionaries in reverse order.
            param_interface.add_custom_params(yaml.load(stream))
        print('Loaded configuration from file {}'.format(config))
        # Add to list of loaded configs.
        configs_to_load.append(config)
    # Done. In here Param Registry contains configuration loaded (and
    # overwritten) from several files.

    # Get problem and model names.
    try:
        task_name = param_interface['training']['problem']['name']
    except BaseException:
        print("Error: Couldn't retrieve problem name from the loaded configuration")
        exit(-1)

    try:
        model_name = param_interface['model']['name']
    except BaseException:
        print("Error: Couldn't retrieve model name from the loaded configuration")
        exit(-1)

    # Prepare output paths for logging
    while True:  # Dirty fix: if log_dir already exists, wait for 1 second and try again
        try:
            time_str = '{0:%Y%m%d_%H%M%S}'.format(datetime.now())
            if FLAGS.savetag != '':
                time_str = time_str + "_" + FLAGS.savetag
            log_dir = FLAGS.outdir + '/' + task_name + \
                '/' + model_name + '/' + time_str + '/'
            os.makedirs(log_dir, exist_ok=False)
        except FileExistsError:
            sleep(1)
        else:
            break

    model_dir = log_dir + 'models/'
    os.makedirs(model_dir, exist_ok=False)
    log_file = log_dir + 'trainer.log'

    # Create tensorboard output - if tensorboard is supposed to be used.
    if FLAGS.tensorboard is not None:
        from tensorboardX import SummaryWriter

        training_writer = SummaryWriter(log_dir + '/training')
        validation_writer = SummaryWriter(log_dir + '/validation')
    else:
        validation_writer = None

    def logfile():
        return logging.FileHandler(log_file)

    # Load default logger configuration.
    with open('logger_config.yaml', 'rt') as f:
        config = yaml.load(f.read())
        logging.config.dictConfig(config)

    # Set logger label and level.
    logger = logging.getLogger('Trainer')
    logger.setLevel(getattr(logging, FLAGS.log.upper(), None))

    # Set random seeds.
    if "seed_torch" not in param_interface["training"] or param_interface["training"]["seed_torch"] == -1:
        seed = randrange(0, 2**32)
        param_interface["training"].add_custom_params({"seed_torch": seed})
    logger.info("Setting torch random seed to: {}".format(
        param_interface["training"]["seed_torch"]))
    torch.manual_seed(param_interface["training"]["seed_torch"])
    torch.cuda.manual_seed_all(param_interface["training"]["seed_torch"])

    if "seed_numpy" not in param_interface["training"] or param_interface["training"]["seed_numpy"] == -1:
        seed = randrange(0, 2**32)
        param_interface["training"].add_custom_params({"seed_numpy": seed})
    logger.info("Setting numpy random seed to: {}".format(
        param_interface["training"]["seed_numpy"]))
    np.random.seed(param_interface["training"]["seed_numpy"])

    # Initialize the application state singleton.
    app_state = AppState()

    # check if CUDA is available turn it on
    check_and_set_cuda(param_interface['training'], logger)

    # Build the model.
    model = ModelFactory.build_model(param_interface['model'])
    model.cuda() if app_state.use_CUDA else None

    # Log the model summary.
    logger.info(model.summarize())

    # Build problem for the training
    problem = ProblemFactory.build_problem(
        param_interface['training']['problem'])

    if 'curriculum_learning' in param_interface['training']:
        # Initialize curriculum learning - with values from config.
        problem.curriculum_learning_initialize(
            param_interface['training']['curriculum_learning'])
        # Set initial values of curriculum learning.
        curric_done = problem.curriculum_learning_update_params(0)

        # If key is not present in config then it has to be finished (DEFAULT:
        # True)
        if 'must_finish' not in param_interface['training'][
                'curriculum_learning']:
            param_interface['training']['curriculum_learning'].add_default_params({
                                                                                  'must_finish': True})
        must_finish_curriculum = param_interface['training'][
            'curriculum_learning']['must_finish']
        logger.info("Using curriculum learning")
    else:
        # Initialize curriculum learning - with empty dict.
        problem.curriculum_learning_initialize({})
        # If not using curriculum then it does not have to be finished.
        must_finish_curriculum = False

    # Model validation interval (DEFAULT: 100).
    try:
        model_validation_interval = param_interface['training'][
            'validation_interval']
    except KeyError:
        model_validation_interval = 100

    # Create statistics collector.
    stat_col = StatisticsCollector()
    # Add model/problem dependent statistics.
    problem.add_statistics(stat_col)
    model.add_statistics(stat_col)

    # Create csv file.
    training_file = stat_col.initialize_csv_file(log_dir, 'training.csv')

    # Check if validation section is present AND problem section is also
    # present...
    if ('validation' in param_interface) and (
            'problem' in param_interface['validation']):
        # ... then load problem, set variables etc.

        # Build problem for the validation
        problem_validation = ProblemFactory.build_problem(
            param_interface['validation']['problem'])
        generator_validation = problem_validation.return_generator()

        # Get a single batch that will be used for validation (!)
        data_valid, aux_valid = next(generator_validation)

        # Create csv file.
        validation_file = stat_col.initialize_csv_file(
            log_dir, 'validation.csv')

        # Turn on validation.
        use_validation_problem = True
        logger.info(
            "Using validation problem for calculation of loss and model validation")

    else:
        # We do not have validation problem - so turn it off.
        use_validation_problem = False
        logger.info(
            "Using training problem for calculation of loss and model validation")

        # Use loss length (DEFAULT: 10).
        try:
            loss_length = param_interface['training']['length_loss']
        except KeyError:
            loss_length = 10

    # Set optimizer.
    optimizer_conf = dict(param_interface['training']['optimizer'])
    optimizer_name = optimizer_conf['name']
    del optimizer_conf['name']
    # Select for optimization only those parameters that require update!
    optimizer = getattr(
        torch.optim,
        optimizer_name)(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        **optimizer_conf)

    # Ok, finished loading the configuration.
    # Save the resulting configuration into a yaml settings file, under log_dir
    with open(log_dir + "training_configuration.yaml", 'w') as yaml_backup_file:
        yaml.dump(param_interface.to_dict(),
                  yaml_backup_file, default_flow_style=False)

    # Log the training configuration.
    conf_str = '\n' + '='*80 + '\n'
    conf_str += 'Final registry configuration for training {} on {}:\n'.format(model_name, task_name)
    conf_str += '='*80 + '\n'
    conf_str += yaml.safe_dump(param_interface.to_dict(), default_flow_style=False)
    conf_str += '='*80 + '\n'
    logger.info(conf_str)

    # Ask for confirmation - optional.
    if FLAGS.confirm:
        # Ask for confirmation
        input('Press any key to continue')

    # Start Training
    episode = 0
    last_losses = collections.deque()

    # Flag denoting whether we converged (or reached last episode).
    terminal_condition = False

    # Main training and verification loop.
    for data_tuple, aux_tuple in problem.return_generator():

        # apply curriculum learning - change problem max seq_length
        curric_done = problem.curriculum_learning_update_params(episode)

        # reset gradients
        optimizer.zero_grad()

        # Check visualization flag - turn on when we wanted to visualize (at
        # least) validation.
        if FLAGS.visualize is not None and FLAGS.visualize <= 1:
            AppState().visualize = True
        else:
            app_state.visualize = False

        # Turn on training mode.
        model.train()
        # 1. Perform forward step, calculate logits and loss.
        logits, loss = forward_step(
            model, problem, episode, stat_col, data_tuple, aux_tuple)

        if not use_validation_problem:
            # Store the calculated loss on a list.
            last_losses.append(loss)
            # Truncate list length.
            if len(last_losses) > loss_length:
                last_losses.popleft()

        # 2. Backward gradient flow.
        loss.backward()
        # Check the presence of parameter 'gradient_clipping'.
        try:
            # if present - clip gradients to a range (-gradient_clipping,
            # gradient_clipping)
            val = param_interface['training']['gradient_clipping']
            nn.utils.clip_grad_value_(model.parameters(), val)
        except KeyError:
            # Else - do nothing.
            pass

        # 3. Perform optimization.
        optimizer.step()

        # 4. Log statistics.
        # Log to logger.
        logger.info(stat_col.export_statistics_to_string())
        # Export to csv.
        stat_col.export_statistics_to_csv(training_file)

        # Export data to tensorboard.
        if (FLAGS.tensorboard is not None) and (
                episode % FLAGS.logging_frequency == 0):
            stat_col.export_statistics_to_tensorboard(training_writer)

            # Export histograms.
            if FLAGS.tensorboard >= 1:
                for name, param in model.named_parameters():
                    try:
                        training_writer.add_histogram(
                            name, param.data.cpu().numpy(), episode, bins='doane')
                    except Exception as e:
                        logger.error("  {} :: data :: {}".format(name, e))
            # Export gradients.
            if FLAGS.tensorboard >= 2:
                for name, param in model.named_parameters():
                    try:
                        training_writer.add_histogram(
                            name + '/grad', param.grad.data.cpu().numpy(), episode, bins='doane')
                    except Exception as e:
                        logger.error("  {} :: grad :: {}".format(name, e))

        # Check visualization of training data.
        if app_state.visualize:
            # Allow for preprocessing
            data_tuple, aux_tuple, logits = problem.plot_preprocessing(
                data_tuple, aux_tuple, logits)
            # Show plot, if user presses Quit - break.
            if model.plot(data_tuple, logits):
                break

        #  5. Validate and (optionally) save the model.
        user_pressed_stop = False
        if (episode % model_validation_interval) == 0:

            # Validate on the problem if required.
            if use_validation_problem:

                # Check visualization flag - turn on when we wanted to
                # visualize (at least) validation.
                if FLAGS.visualize is not None and (
                        FLAGS.visualize == 1 or FLAGS.visualize == 2):
                    app_state.visualize = True
                else:
                    app_state.visualize = False

                # Perform validation.
                validation_loss, user_pressed_stop = validation(
                    model, problem, episode, stat_col, data_valid, aux_valid,
                    FLAGS, logger, validation_file, validation_writer)

            # Save the model using latest (validation or training) statistics.
            model.save(model_dir, stat_col)

        # 6. Terminal conditions.
        # I. User pressed stop during visualization.
        if user_pressed_stop:
            break

        # II. & III - only when we finished curriculum.
        if curric_done or not must_finish_curriculum:
            # break if conditions applied: convergence or max episodes
            loss_stop = False
            if use_validation_problem:
                loss_stop = validation_loss < param_interface['training'][
                    'terminal_condition']['loss_stop']
                # We already saved that model.
            else:
                loss_stop = max(last_losses) < param_interface['training'][
                    'terminal_condition']['loss_stop']
                # We already saved that model.

            if loss_stop:
                # Ok, we have converged.
                terminal_condition = True
                # "Finish" the training.
                break

        if episode == param_interface['training']['terminal_condition'][
                'max_episodes']:
            terminal_condition = True
            # If we are here then it means that we didn't converged and the model is bad for sure.
            # But let's try to save it anyway, maybe it is still better than
            # the previous one.

            # Validate on the problem if required - so we can collect the
            # statistics needed during saving of the best model.
            if use_validation_problem:
                # Perform validation.
                validation_loss, user_pressed_stop = validation(
                    model, problem, episode, stat_col, data_valid, aux_valid,
                    FLAGS, logger, validation_file, validation_writer)

            model.save(model_dir, stat_col)
            # "Finish" the training.
            break

        # Next episode.
        episode += 1

    # Check whether we have finished training properly.
    if terminal_condition:
        logger.info('Learning finished!')
        # Check visualization flag - turn on when we wanted to visualize (at
        # least) validation.
        if FLAGS.visualize is not None and (FLAGS.visualize == 3):
            app_state.visualize = True

            # Perform validation.
            if use_validation_problem:
                _, _ = validation(
                    model, problem, episode, stat_col, data_valid, aux_valid,
                    FLAGS, logger, validation_file, validation_writer)

        else:
            app_state.visualize = False

    else:
        logger.warning('Learning interrupted!')

    # Close files.
    training_file.close()
    validation_file.close()
    if (FLAGS.tensorboard is not None):
        # Close TB writers.
        training_writer.close()
        validation_writer.close()
