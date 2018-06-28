# Force MKL (CPU BLAS) to use one core, faster
import logging
import logging.config
import os

os.environ["OMP_NUM_THREADS"] = '1'

import yaml
import os.path
from datetime import datetime
from time import sleep
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import collections
import numpy as np

from misc.app_state import AppState
from misc.statistics_collector import StatisticsCollector

# Import model factory.
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.model_factory import ModelFactory
from misc.param_interface import ParamInterface

# Import problems factory and data tuple.
from problems.problem_factory import ProblemFactory
from utils_worker import forward_step, check_and_set_cuda


def save_model(model, episode,   model_dir):
    """
    Function saves the model..

    :returns: False if saving was successful (need to implement true condition if there was an error)
    """
    model_filename = 'model_parameters_episode_{:05d}'.format(episode)
    torch.save(model.state_dict(), model_dir + model_filename)
    logger.info("Model exported")


def validation(model, problem, episode, stat_col, data_valid, aux_valid,  FLAGS, logger, validation_file,
               validation_writer):
    """
    Function performs validation of the model, using the provided data and criterion.
    Additionally it logs (to files, tensorboard) and visualizes.

    :param stat_col: Statistic collector object.
    :return: True if training loop is supposed to end.
    """
    model.eval()
    # Calculate loss of the validation data.
    with torch.no_grad():
        logits_valid, loss_valid = forward_step(model, problem, episode, stat_col, data_valid, aux_valid)

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
        return loss_valid, model.plot(data_valid,  logits_valid)
    # Else simply return false, i.e. continue training.
    return loss_valid, False


def curriculum_learning_update_problem_params(problem, episode, param_interface):
    """
    Updates problem parameters according to curriculum learning.

    :returns: Boolean informing whether curriculum learning is finished (or wasn't active at all).
    """
    # Curriculum learning stop condition.
    curric_done = True
    try:
        # If the 'curriculum_learning' section is not present, this line will throw an exception.
        curr_config = param_interface['problem_train']['curriculum_learning']

        # Read min and max length.
        min_length = param_interface['problem_train']['min_sequence_length']
        max_max_length = param_interface['problem_train']['max_sequence_length']

        if curr_config['interval'] > 0:
            # Curriculum learning goes from the initial max length to the max length in steps of size 1
            max_length = curr_config['initial_max_sequence_length'] + (episode // curr_config['interval'])
            if max_length >= max_max_length:
                max_length = max_max_length
            else:
                curric_done = False
                # Change max length.
            problem.set_max_length(max_length)
    except KeyError:
        pass
    # Return information whether we finished CL (i.e. reached max sequence length).
    return curric_done


if __name__ == '__main__':
    # Create parser with list of  runtime arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--agree', dest='confirm', action='store_true',
                        help='Request user confirmation just after loading the settings, before starting training  (Default: False)')
    parser.add_argument('--config', dest='config', type=str, default='',
                        help='Name of the configuration file to be loaded')
    parser.add_argument('--savetag', dest='savetag', type=str, default='',
                        help='Tag for the save directory')
    parser.add_argument('--tensorboard', action='store', dest='tensorboard', choices=[0, 1, 2], type=int,
                        help="If present, log to TensorBoard. Log levels:\n"
                             "0: Just log the loss, accuracy, and seq_len\n"
                             "1: Add histograms of biases and weights (Warning: slow)\n"
                             "2: Add histograms of biases and weights gradients (Warning: even slower)")
    parser.add_argument('--lf', dest='logging_frequency', default=100, type=int,
                        help='TensorBoard logging frequency (Default: 100, i.e. logs every 100 episodes)')
    parser.add_argument('--log', action='store', dest='log', type=str, default='INFO',
                        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                        help="Log level. (Default: INFO)")
    parser.add_argument('--visualize', dest='visualize', choices=[0, 1, 2, 3], type=int,
                        help="Activate dynamic visualization:\n"
                             "0: Only during training\n"
                             "1: During both training and validation\n"
                             "2: Only during validation\n"
                             "3: Only during last validation, after training is completed\n")

    # Parse arguments.
    FLAGS, unparsed = parser.parse_known_args()

    # Check if config file was selected.
    if FLAGS.config == '':
        print('Please pass configuration file as --c parameter')
        exit(-1)
    # Check if file exists.
    if not os.path.isfile(FLAGS.config):
        print('Configuration file {} does not exists'.format(FLAGS.config))
        exit(-2)

    # Read the YAML file.
    param_interface = ParamInterface()
    with open(FLAGS.config, 'r') as stream:
        param_interface.add_custom_params(yaml.load(stream))

    # Get problem and model names.
    task_name = param_interface['problem_train']['name']
    model_name = param_interface['model']['name']


    # Prepare output paths for logging
    path_root = "./experiments/"
    while True:  # Dirty fix: if log_dir already exists, wait for 1 second and try again
        try:
            time_str = '{0:%Y%m%d_%H%M%S}'.format(datetime.now())
            if FLAGS.savetag != '':
                time_str = time_str + "_" + FLAGS.savetag
            log_dir = path_root + task_name + '/' + model_name + '/' + time_str + '/'
            os.makedirs(log_dir, exist_ok=False)
        except FileExistsError:
            sleep(1)
        else:
            break

    model_dir = log_dir + 'models/'
    os.makedirs(model_dir, exist_ok=False)
    log_file = log_dir + 'msgs.log'

    # Copy the training config into a yaml settings file, under log_dir
    with open(log_dir + "/train_settings.yaml", 'w') as yaml_backup_file:
        yaml.dump(param_interface.to_dict(), yaml_backup_file, default_flow_style=False)

    # Create tensorboard output - if tensorboard is supposed to be used.
    if FLAGS.tensorboard is not None:
        from tensorboardX import SummaryWriter

        training_writer = SummaryWriter(log_dir + '/training')
        validation_writer = SummaryWriter(log_dir + '/validation')
    else:
        validation_writer = None


    def logfile():
        return logging.FileHandler(log_file)


    # Log configuration to file.
    with open('logger_config.yaml', 'rt') as f:
        config = yaml.load(f.read())
        logging.config.dictConfig(config)

    # Set logger label and level.
    logger = logging.getLogger('Train')
    logger.setLevel(getattr(logging, FLAGS.log.upper(), None))

    # Print experiment configuration
    str = 'Configuration for {}:\n'.format(task_name)
    str += yaml.safe_dump(param_interface.to_dict(), default_flow_style=False,
                          explicit_start=True, explicit_end=True)
    logger.info(str)

    # Ask for confirmation - optional.
    if FLAGS.confirm:
        # Ask for confirmation
        input('Press any key to continue')

    # Set random seeds.
    if param_interface["settings"]["seed_torch"] != -1:
        torch.manual_seed(param_interface["settings"]["seed_torch"])
        torch.cuda.manual_seed_all(param_interface["settings"]["seed_torch"])

    if param_interface["settings"]["seed_numpy"] != -1:
        np.random.seed(param_interface["settings"]["seed_numpy"])

        # Initialize the application state singleton.
    app_state = AppState()
    # If we are going to use SOME visualization - set flag to True now, before creation of problem and model objects.
    if FLAGS.visualize is not None:
        app_state.visualize = True

    # check if CUDA is available turn it on
    check_and_set_cuda(param_interface['problem_train'], logger) 

    # Build problem for the training
    problem = ProblemFactory.build_problem(param_interface['problem_train'])

    # Initialize curriculum learning.
    curric_done = curriculum_learning_update_problem_params(problem, 0, param_interface)

    # Build problem for the validation
    problem_validation = ProblemFactory.build_problem(param_interface['problem_validation'])
    generator_validation = problem_validation.return_generator()

    # Get a single batch that will be used for validation (!)
    data_valid, aux_valid = next(generator_validation)

    # Build the model.
    model = ModelFactory.build_model(param_interface['model'])
    model.cuda() if app_state.use_CUDA else None

    # Set optimizer.
    optimizer_conf = dict(param_interface['optimizer'])
    optimizer_name = optimizer_conf['name']
    del optimizer_conf['name']
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_conf)

    # Start Training
    episode = 0
    last_losses = collections.deque()
    validation_loss=.7 #default value so the loop won't terminate if the validation is not done on the first step

    # Create statistics collector.
    stat_col = StatisticsCollector()
    # Add model/problem dependent statistics.
    problem.add_statistics(stat_col)
    model.add_statistics(stat_col)

    # Create csv files.
    train_file = stat_col.initialize_csv_file(log_dir, '/training.csv')
    validation_file = stat_col.initialize_csv_file(log_dir, '/validation.csv')

    # Try to read validation frequency from config, else set default (100)
    try:
        validation_frequency = param_interface['problem_validation']['frequency']
    except KeyError:
        validation_frequency = 100

    #whether to do validation (default True)
    try:
        do_validation = param_interface['settings']['do_validation']
    except KeyError:
        do_validation = True

    if do_validation:
    # Figure out if validation is defined else assume that it should be true
        try:
            validation_stopping = param_interface['settings']['validation_stopping']
        except KeyError:
            validation_stopping = True
    else:
        validation_stopping = False

    # Flag denoting whether we converged (or reached last episode).
    terminal_condition = False

    # Main training and verification loop.
    for data_tuple, aux_tuple in problem.return_generator():

        # apply curriculum learning - change problem max seq_length
        curric_done = curriculum_learning_update_problem_params(problem, episode, param_interface)

        # reset gradients
        optimizer.zero_grad()

        # Check visualization flag - turn on when we wanted to visualize (at least) validation.
        if FLAGS.visualize is not None and FLAGS.visualize <= 1:
            AppState().visualize = True
        else:
            app_state.visualize = False

        model.train()
        # 1. Perform forward step, calculate logits and loss.
        logits, loss = forward_step(model, problem, episode, stat_col, data_tuple, aux_tuple)

        # Store the calculated loss on a list.
        last_losses.append(loss)
        # Truncate list length.
        if len(last_losses) > param_interface['settings']['length_loss']:
            last_losses.popleft()

        # 2. Backward gradient flow.
        loss.backward()
        # Check the presence of parameter 'gradient_clipping'.
        try:
            # if present - clip gradients to a range (-gradient_clipping, gradient_clipping)
            val = param_interface['problem_train']['gradient_clipping']
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
        stat_col.export_statistics_to_csv(train_file)

        # Export data to tensorboard.
        if (FLAGS.tensorboard is not None) and (episode % FLAGS.logging_frequency == 0):
            stat_col.export_statistics_to_tensorboard(training_writer)

            # Export histograms.
            if FLAGS.tensorboard >= 1:
                for name, param in model.named_parameters():
                    try:
                        training_writer.add_histogram(name, param.data.cpu().numpy(), episode, bins='doane')
                    except Exception as e:
                        logger.error("  {} :: data :: {}".format(name, e))
            # Export gradients.
            if FLAGS.tensorboard >= 2:
                for name, param in model.named_parameters():
                    try:
                        training_writer.add_histogram(name + '/grad', param.grad.data.cpu().numpy(), episode, bins='doane')
                    except Exception as e:
                        logger.error("  {} :: grad :: {}".format(name, e))

        # Check visualization of training data.
        if app_state.visualize:
            # Show plot, if user presses Quit - break.
            if model.plot(data_tuple,  logits):
                break

        #  5. Save the model then validate
        if (episode % validation_frequency) == 0:
            save_model(model, episode,  model_dir)


        if (episode % validation_frequency) == 0 and do_validation:

            # Check visualization flag - turn on when we wanted to visualize (at least) validation.
            if FLAGS.visualize is not None and (FLAGS.visualize == 1 or FLAGS.visualize == 2):
                app_state.visualize = True
            else:
                app_state.visualize = False

            validation_loss, stop_now = validation(model, problem, episode, stat_col, data_valid, aux_valid,  FLAGS,
                    logger,   validation_file,  validation_writer)
            
            # Perform validation.
            if stop_now:
                break
            # End of validation.


        if curric_done:
            # break if conditions applied: convergence or max episodes
            loss_stop=True
            if validation_stopping:
                loss_stop = validation_loss < param_interface['settings']['loss_stop']
            else:
                loss_stop = max(last_losses) < param_interface['settings']['loss_stop']

            if loss_stop or episode == param_interface['settings']['max_episodes'] :
                terminal_condition = True
                save_model(model, episode,  model_dir)

                break
                # "Finish" episode.

        episode += 1

    # Check whether we have finished training!
    if terminal_condition:
        logger.info('Learning finished!')
        logger.info('Model saved in '+ log_dir)
        # Check visualization flag - turn on when we wanted to visualize (at least) validation.
        if FLAGS.visualize is not None and (FLAGS.visualize == 3):
            app_state.visualize = True

            # Perform validation.
            _, _ = validation(model, problem, episode, stat_col, data_valid, aux_valid, FLAGS, logger,
                               validation_file, validation_writer, stat_col)

        else:
            app_state.visualize = False


    else:
        logger.info('Learning interrupted!')

    # Close files.
    train_file.close()
    validation_file.close()
    if (FLAGS.tensorboard is not None):
        # Close TB writers.
        training_writer.close()
        validation_writer.close()
