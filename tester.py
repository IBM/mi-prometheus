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

"""tester.py: contains code of worker realising testing of previously trained models"""
__author__= "Alexis Asseman, Ryan McAvoy, Tomasz Kornuta"

import os
# Force MKL (CPU BLAS) to use one core, faster
os.environ["OMP_NUM_THREADS"] = '1'

import yaml
from random import randrange

from datetime import datetime
from time import sleep

import torch
import argparse

import numpy as np
from glob import glob

import logging
import logging.config

# Import problems and problem factory.
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'problems'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from problems.problem_factory import ProblemFactory
from models.model_factory import ModelFactory

from misc.app_state import AppState
from misc.statistics_collector import StatisticsCollector
from misc.param_interface import ParamInterface
from worker_utils import forward_step, check_and_set_cuda

logging.getLogger('matplotlib').setLevel(logging.WARNING)


if __name__ == '__main__':
    # Create parser with list of  runtime arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', dest='model',
                        help='Path to and name of the file containing the saved parameters of the model (model checkpoint)')
    parser.add_argument('--savetag', dest='savetag', type=str, default='',
                        help='Tag for the save directory')
    parser.add_argument('--log', action='store', dest='log', type=str, default='info',
                        choices=['critical', 'error', 'warning', 'info', 'debug', 'notset'],
                        help="Log level. Default is INFO.")
    parser.add_argument('--visualize', action='store_true', dest='visualize',
                        help='Activate dynamic visualization')

    # Parse arguments.
    FLAGS, unparsed = parser.parse_known_args()
    
    # Check if model is present.
    if FLAGS.model == '':
        print('Please pass path to and name of the file containing model to be loaded as --m parameter')
        exit(-1)

    # Check if file with model exists.
    if not os.path.isfile(FLAGS.model):
        print('Model file {} does not exist'.format(FLAGS.model))
        exit(-2)

    # Extract path.
    abs_path, model_dir = os.path.split(os.path.dirname(os.path.abspath(FLAGS.model)))

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
            if FLAGS.savetag != '':
                time_str = time_str + "_" + FLAGS.savetag
            log_dir = abs_path + '/' + time_str + '/'
            os.makedirs(log_dir, exist_ok=False)
        except FileExistsError:
            sleep(1)
        else:
            break

    # Logging - to subdir
    log_file = log_dir + 'tester.log'
    def logfile():
        return logging.FileHandler(log_file)

    # Load default logger configuration.
    with open('logger_config.yaml', 'rt') as f:
        config = yaml.load(f.read())
        logging.config.dictConfig(config)

    logger = logging.getLogger('Tester')
    logger.setLevel(getattr(logging, FLAGS.log.upper(), None))

    # Initialize the application state singleton.
    app_state = AppState()

    if FLAGS.visualize:
        app_state.visualize = True

    # Initialize parameter interface.
    param_interface = ParamInterface()

    # Read YAML file    
    with open(config_file, 'r') as stream:
        param_interface.add_custom_params(yaml.load(stream))

    # Set random seeds.
    if "seed_torch" not in param_interface["testing"] or param_interface["testing"]["seed_torch"] == -1:
        seed = randrange(0, 2**32)
        param_interface["testing"].add_custom_params({"seed_torch": seed})
    logger.info("Setting torch random seed to: {}".format(param_interface["testing"]["seed_torch"]))
    torch.manual_seed(param_interface["testing"]["seed_torch"])
    torch.cuda.manual_seed_all(param_interface["testing"]["seed_torch"])

    if "seed_numpy" not in param_interface["testing"] or param_interface["testing"]["seed_numpy"] == -1:
        seed = randrange(0, 2**32)
        param_interface["testing"].add_custom_params({"seed_numpy": seed})
    logger.info("Setting numpy random seed to: {}".format(param_interface["testing"]["seed_numpy"]))
    np.random.seed(param_interface["testing"]["seed_numpy"])

    # Initialize the application state singleton.
    app_state = AppState()

    # check if CUDA is available turn it on
    check_and_set_cuda(param_interface['testing'], logger) 

    # Get problem and model names.
    try:
        task_name = param_interface['testing']['problem']['name']
    except:
        print("Error: Couldn't retrieve the problem name from the loaded configuration")
        exit(-1)

    try:
        model_name = param_interface['model']['name']
    except:
        print("Error: Couldn't retrieve model name from the loaded configuration")
        exit(-1)

    # Create model object.
    model = ModelFactory.build_model(param_interface['model'])
    model.cuda() if app_state.use_CUDA else None
    # Load parameters from checkpoint.
    model.load(FLAGS.model)
    # Turn on evaluation mode.
    model.eval()

    # Build problem.
    problem = ProblemFactory.build_problem(param_interface['testing']['problem'])

    # Create statistics collector.
    stat_col = StatisticsCollector()
    # Add model/problem dependent statistics.
    problem.add_statistics(stat_col)
    model.add_statistics(stat_col)

    # Create test output csv file.
    test_file = stat_col.initialize_csv_file(log_dir, 'testing.csv')

    # Ok, finished loading the configuration. 
    # Save the resulting configuration into a yaml settings file, under log_dir
    with open(log_dir + "testing_configuration.yaml", 'w') as yaml_backup_file:
        yaml.dump(param_interface.to_dict(), yaml_backup_file, default_flow_style=False)

    # Run test
    with torch.no_grad():
        for episode, (data_tuple, aux_tuple) in enumerate(problem.return_generator()):

            logits, loss = forward_step(model, problem, episode, stat_col, data_tuple, aux_tuple)

            # Log to logger.
            logger.info(stat_col.export_statistics_to_string('[Test]'))
            # Export to csv.
            stat_col.export_statistics_to_csv(test_file)

            if app_state.visualize:
                # Allow for preprocessing
                data_tuple, aux_tuple, logits = problem.plot_preprocessing(data_tuple, aux_tuple, logits)

                # Show plot, if user presses Quit - break.
                is_closed = model.plot(data_tuple,  logits)
                if is_closed:
                    break
            elif episode == param_interface["testing"]["problem"]["max_test_episodes"]:
                break

