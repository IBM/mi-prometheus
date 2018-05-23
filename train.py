# Force MKL (CPU BLAS) to use one core, faster
import logging
import logging.config
import os

os.environ["OMP_NUM_THREADS"] = '1'

import yaml
import os.path
from shutil import copyfile
from datetime import datetime
from time import sleep
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import collections
import numpy as np

from misc.app_state import AppState

# Import model factory.
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.model_factory import ModelFactory

# Import problems factory and data tuple.
sys.path.append(os.path.join(os.path.dirname(__file__), 'problems'))
from problems.problem_factory import ProblemFactory
from problems.algorithmic_sequential_problem import DataTuple

use_CUDA = False

def forward_step(model, data_tuple,  use_mask,  criterion):
    """ Function performs a single forward step.

    :returns: logits, loss and accuracy (former using provided criterion)
    """
    # Unpack the data tuple.
    (inputs, targets, mask) = data_tuple

    if use_CUDA:
        inputs = inputs.cuda()
        targets = targets.cuda()
        mask = mask.cuda()

    # 1. Perform forward calculation.
    logits = model(inputs)

    # 2. Calculate loss.
    # Check if mask should be is used - if so, apply.
    if use_mask:
        masked_logits = logits[:, mask[0], :]
        masked_targets = targets[:, mask[0], :]
    else:
        masked_logits = logits
        masked_targets = targets

    # Compute loss using the provided criterion.
    loss = criterion(masked_logits, masked_targets)
    # Calculate accuracy.
    accuracy = (1 - torch.abs(torch.round(F.sigmoid(masked_logits)) - masked_targets)).mean()
    # Return tuple: logits, loss, accuracy.
    return logits, loss, accuracy

def save_model(model, episode,   model_dir): 
    """
    Function saves the model..
    
    :returns: False if saving was successful (need to implement true condition if there was an error)
    """
    model_filename = 'model_parameters_episode_{:05d}'.format(episode)
    torch.save(model.state_dict(), model_dir + model_filename)
    logger.info("Model exported")

def validation(model, data_valid, use_mask, criterion, FLAGS, logger, validation_file,
               validation_writer):
    """
    Function performs validation of the model, using the provided data and criterion.
    Additionally it logs (to files, tensorboard) and visualizes.
    
    :returns: True if training loop is supposed to end.
    """
 
    # Calculate the accuracy and loss of the validation data.
    with torch.no_grad():
        logits_valid, loss_valid, accuracy_valid = forward_step(model, data_valid, use_mask, criterion)

    # Print statistics.
    length_valid = data_valid[0].size(-2)
    format_str = 'episode {:05d}; acc={:12.10f}; loss={:12.10f}; length={:d} [Validation]'

    logger.info(format_str.format(episode, accuracy_valid, loss_valid, length_valid))
    format_str = '{:05d}, {:12.10f}, {:12.10f}, {:03d}'

    format_str = format_str + '\n'
    validation_file.write(format_str.format(episode, accuracy_valid, loss_valid, length_valid))

    if (FLAGS.tensorboard is not None):
        # Save loss + accuracy to tensorboard
        validation_writer.add_scalar('Loss', loss_valid, episode)
        validation_writer.add_scalar('Accuracy', accuracy_valid, episode)
        validation_writer.add_scalar('Seq_len', length_valid, episode)

    # Visualization of validation.
    if AppState().visualize:
        (inputs_valid, targets_valid, _) = data_valid
        # True means that we should terminate
        return loss_valid,  model.plot_sequence(inputs_valid[0].detach(), logits_valid[0].detach(), targets_valid[0].detach())
    # Else simply return false, i.e. continue training.
    return loss_valid, False


def curriculum_learning_update_problem_params(problem, episode, config_loaded):
    """
    Updates problem parameters according to curriculum learning.
    
    :returns: Boolean informing whether curriculum learning is finished (or wasn't active at all).
    """
    # Curriculum learning stop condition.
    curric_done = True
    try:
        # If the 'curriculum_learning' section is not present, this line will throw an exception.
        curr_config = config_loaded['problem_train']['curriculum_learning']

        # Read min and max length.
        min_length = config_loaded['problem_train']['min_sequence_length']
        max_max_length = config_loaded['problem_train']['max_sequence_length']

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
    parser.add_argument('--confirm', dest='confirm', action='store_true',
                        help='Request user confirmation just after loading the settings, before starting training  (Default: False)')
    parser.add_argument('-t', dest='task', type=str, default='',
                        help='Name of the task configuration file to be loaded')
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
    parser.add_argument('-v', dest='visualize', choices=[0, 1, 2, 3], type=int,
                        help="Activate dynamic visualization:\n"
                             "0: Only during training\n"
                             "1: During both training and validation\n"
                             "2: Only during validation\n"
                             "3: Only during last validation, after training is completed\n")

    # Parse arguments.
    FLAGS, unparsed = parser.parse_known_args()

    # Check if config file was selected.
    if FLAGS.task == '':
        print('Please pass task configuration file as -t parameter')
        exit(-1)
    # Check if file exists.
    if not os.path.isfile(FLAGS.task):
        print('Task configuration file {} does not exists'.format(FLAGS.task))
        exit(-2)

    # Read the YAML file.
    with open(FLAGS.task, 'r') as stream:
        config_loaded = yaml.load(stream)

    # Get problem and model names.
    task_name = config_loaded['problem_train']['name']
    model_name = config_loaded['model']['name']

    # Prepare output paths for logging
    path_root = "./checkpoints/"
    while True:  # Dirty fix: if log_dir already exists, wait for 1 second and try again
        try:
            time_str = '{0:%Y%m%d_%H%M%S}'.format(datetime.now())
            log_dir = path_root + task_name + '/' + model_name + '/' + time_str + '/'
            os.makedirs(log_dir, exist_ok=False)
        except FileExistsError:
            sleep(1)
        else:
            break

    model_dir = log_dir + 'models/'
    os.makedirs(model_dir, exist_ok=False)
    log_file = log_dir + 'msgs.log'
    copyfile(FLAGS.task, log_dir + "/train_settings.yaml")  # Copy the task's yaml file into log_dir

    # Create csv files.
    train_file = open(log_dir + 'training.csv', 'w', 1)
    validation_file = open(log_dir + 'validation.csv', 'w', 1)
    train_file.write('episode,accuracy,loss,length\n')
    validation_file.write('episode,accuracy,length\n')

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
    str += yaml.safe_dump(config_loaded, default_flow_style=False,
                          explicit_start=True, explicit_end=True)
    logger.info(str)

    # Ask for confirmation - optional.
    if FLAGS.confirm:
        # Ask for confirmation
        input('Press any key to continue')

    # Set random seeds.
    if config_loaded["settings"]["seed_torch"] != -1:
        torch.manual_seed(config_loaded["settings"]["seed_torch"])
        torch.cuda.manual_seed_all(config_loaded["settings"]["seed_torch"])

    if config_loaded["settings"]["seed_numpy"] != -1:
        np.random.seed(config_loaded["settings"]["seed_numpy"])

    # Determine if CUDA is to be used.
    if torch.cuda.is_available():
        try:  # If the 'cuda' key is not present, catch the exception and do nothing
            if config_loaded['problem_train']['cuda']:
                use_CUDA = True
        except KeyError:
            pass

    # Initialize the application state singleton.
    app_state = AppState()
    # If we are going to use SOME visualization - set flag to True now, before creation of problem and model objects.
    if FLAGS.visualize is not None:
        app_state.visualize = True

    # Build problem for the training
    problem = ProblemFactory.build_problem(config_loaded['problem_train'])

    # Initialize curriculum learning.
    curric_done = curriculum_learning_update_problem_params(problem, 0, config_loaded)

    # Build problem for the validation
    problem_validation = ProblemFactory.build_problem(config_loaded['problem_validation'])
    generator_validation = problem_validation.return_generator()

    # Get a single batch that will be used for validation (!)
    data_valid = next(generator_validation)

    # Build the model.
    model = ModelFactory.build_model(config_loaded['model'])
    model.cuda() if use_CUDA else None

    # Set optimizer.
    optimizer_conf = dict(config_loaded['optimizer'])
    optimizer_name = optimizer_conf['name']
    del optimizer_conf['name']
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_conf)

    # Set loss criterion.
    # TK: TODO: move criterion to PROBLEM!
    criterion = nn.BCEWithLogitsLoss()

    # Start Training
    episode = 0
    last_losses = collections.deque()
    validation_loss=.7 #default value so the loop won't terminate if the validation is not done on the first step

    # Try to read validation frequency from config, else set default (100)
    try:
        validation_frequency = config_loaded['problem_validation']['frequency']
    except KeyError:
        validation_frequency = 100

    #whether to do validation (default True)
    try: 
        do_validation = config_loaded['settings']['do_validation']
    except KeyError:
        do_validation = True
 
    if do_validation:
    # Figure out if validation is defined else assume that it should be true
        try: 
            validation_stopping = config_loaded['settings']['validation_stopping']
        except KeyError:
            validation_stopping = True
    else:
        validation_stopping = False

    # Flag denoting whether we converged (or reached last episode).
    terminal_condition = False

    # Main training and verification loop.
    for inputs, targets, mask in problem.return_generator():
        # Convert inputs and targets to CUDA
        if use_CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # apply curriculum learning - change problem max seq_length
        curric_done = curriculum_learning_update_problem_params(problem, episode, config_loaded)

        # reset gradients
        optimizer.zero_grad()

        # Check visualization flag - turn on when we wanted to visualize (at least) validation.
        if FLAGS.visualize is not None and FLAGS.visualize <= 1:
            AppState().visualize = True
        else:
            app_state.visualize = False

        # 1. Perform forward step, calculate logits, loss  and accuracy.
        logits, loss, accuracy = forward_step(model, DataTuple(inputs, targets, mask),
                                              config_loaded['settings']['use_mask'], criterion)

        # Store the calculated loss on a list.
        last_losses.append(loss)
        # Truncate list length.
        if len(last_losses) > config_loaded['settings']['length_loss']:
            last_losses.popleft()

        # 2. Backward gradient flow.
        loss.backward()
        # Check the presence of parameter 'gradient_clipping'.
        try:
            # if present - clip gradients to a range (-gradient_clipping, gradient_clipping)
            val = config_loaded['problem_train']['gradient_clipping']
            nn.utils.clip_grad_value_(model.parameters(), val)
        except KeyError:
            # Else - do nothing.
            pass

        # 3. Perform optimization.
        optimizer.step()

        # 4. Log data - loss, accuracy and other variables (seq_length).
        train_length = inputs.size(-2)
        format_str = 'episode {:05d}; acc={:12.10f}; loss={:12.10f}; length={:d}'
        logger.info(format_str.format(episode, accuracy, loss, train_length))
        format_str = '{:05d}, {:12.10f}, {:12.10f}, {:02d}\n'
        train_file.write(format_str.format(episode, accuracy, loss, train_length))

        # Export data to tensorboard.
        if (FLAGS.tensorboard is not None) and (episode % FLAGS.logging_frequency == 0):
            training_writer.add_scalar('Loss', loss, episode)
            training_writer.add_scalar('Accuracy', accuracy, episode)
            training_writer.add_scalar('Seq_len', train_length, episode)

            # Export histograms.
            if FLAGS.tensorboard >= 1:
                for name, param in model.named_parameters():
                    training_writer.add_histogram(name, param.data.cpu().numpy(), episode, bins='doane')

            # Export gradients.
            if FLAGS.tensorboard >= 2:
                for name, param in model.named_parameters():
                    training_writer.add_histogram(name + '/grad', param.grad.data.cpu().numpy(), episode, bins='doane')

        # Check visualization of training data.
        if app_state.visualize:
            # Show plot, if user presses Quit - break.
            if model.plot_sequence(inputs[0].detach(), logits[0].detach(), targets[0].detach()):
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

            validation_loss, stop_now = validation(model, data_valid,  config_loaded['settings']['use_mask'],  criterion,  FLAGS, 
                    logger,   validation_file,  validation_writer) 
            # Perform validation.
            if stop_now:
                break
            # End of validation.
        

        if curric_done:
            # break if conditions applied: convergence or max episodes
            loss_stop=True
            if validation_stopping:
                loss_stop = validation_loss < config_loaded['settings']['loss_stop']
            else:
                loss_stop = max(last_losses) < config_loaded['settings']['loss_stop']

            if loss_stop or episode == config_loaded['settings']['max_episodes'] :
                terminal_condition = True
                save_model(model, episode,  model_dir)
                    
                break
                # "Finish" episode.

        episode += 1

    # Check whether we have finished training!
    if terminal_condition:
        logger.info('Learning finished!')
        # Check visualization flag - turn on when we wanted to visualize (at least) validation.
        if FLAGS.visualize is not None and (FLAGS.visualize == 3):
            app_state.visualize = True

            # Perform validation.
            _, _ = validation(model, data_valid, config_loaded['settings']['use_mask'], criterion, FLAGS, logger,
                               validation_file, validation_writer)

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
