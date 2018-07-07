# Force MKL (CPU BLAS) to use one core, faster
import os
os.environ["OMP_NUM_THREADS"] = '1'

import torch
import argparse
import yaml
import numpy as np
from glob import glob

import logging
import logging.config

# Force MKL (CPU BLAS) to use one core, faster
os.environ["OMP_NUM_THREADS"] = '1'

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
    parser.add_argument('--visualize', action='store_true', dest='visualize',
                        help='Activate dynamic visualization')
    parser.add_argument('--log', action='store', dest='log', type=str, default='info',
                        choices=['critical', 'error', 'warning', 'info', 'debug', 'notset'],
                        help="Log level. Default is INFO.")

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
    config_file = abs_path + '/train_settings.yaml'
    if not os.path.isfile(config_file):
        print('Config file {} does not exist'.format(config_file))
        exit(-3)

    # Logging - to the same dir. :]
    log_file = abs_path + '/msgs_test.log'
    def logfile():
        return logging.FileHandler(log_file)

    with open('logger_config.yaml', 'rt') as f:
        config = yaml.load(f.read())
        logging.config.dictConfig(config)

    logger = logging.getLogger('Test')
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

    # Set seeds.
    if param_interface["settings"]["seed_torch"] != -1:
        torch.manual_seed(param_interface["settings"]["seed_torch"])
        torch.cuda.manual_seed_all(param_interface["settings"]["seed_torch"])

    if param_interface["settings"]["seed_numpy"] != -1:
        np.random.seed(param_interface["settings"]["seed_numpy"])

    # Check is CUDA is available and turn it on
    check_and_set_cuda(param_interface['problem_test'], logger)

    # Build problem.
    problem = ProblemFactory.build_problem(param_interface['problem_test'])

    # Create model object and load parameters from checkpoint.
    model = ModelFactory.build_model(param_interface['model'])
    model.cuda() if app_state.use_CUDA else None
    model.load(FLAGS.model)

    # Create statistics collector.
    stat_col = StatisticsCollector()
    # Add model/problem dependent statistics.
    problem.add_statistics(stat_col)
    model.add_statistics(stat_col)

    # Create test output csv file.
    test_file = stat_col.initialize_csv_file(abs_path, '/test.csv')

    # Run test
    with torch.no_grad():
        for episode, (data_tuple, aux_tuple)  in enumerate(problem.return_generator()):

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
            else:
                break
