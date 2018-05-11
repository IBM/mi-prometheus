# Force MKL (CPU BLAS) to use one core, faster
import logging
import logging.config
import os
os.environ["OMP_NUM_THREADS"] = '1'

import yaml
import os.path
from shutil import copyfile
from datetime import datetime
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import collections
import numpy as np


# Import model factory.
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.model_factory import ModelFactory

# Import problems factory and data tuple.
sys.path.append(os.path.join(os.path.dirname(__file__), 'problems'))
from problems.problem_factory import ProblemFactory
from problems.algorithmic_sequential_problem import DataTuple

def forward_step(model, data_tuple,  use_mask,  criterion):
    """ Function performs a single forward step.

    :returns: logits, loss and accuracy (former using provided criterion)
    """
    # Unpack the data tuple.
    (inputs, targets, mask) = data_tuple

    # 1. Perform forward calculation.
    logits = model(inputs)

    # 2. Calculate loss.
    # Check if mask should be is used - if so, apply.
    if use_mask:
        logits = logits[:, mask[0], :]
        targets = targets[:, mask[0], :]

    # Compute loss using the provided criterion.
    loss = criterion(logits, targets)
    # Calculate accuracy.
    accuracy = (1 - torch.abs(torch.round(F.sigmoid(logits)) - targets)).mean()
    # Return tuple: logits, loss, accuracy.
    return logits,  loss, accuracy


if __name__ == '__main__':

    # Create parser with list of  runtime arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--confirm', dest='confirm', action='store_true',
                        help='Request user confirmation just after loading the settings, before starting training  (Default: False)')
    parser.add_argument('-t', dest='task',  type=str, default='',
                        help='Name of the task configuration file to be loaded')
    parser.add_argument('--tensorboard', action='store', dest='tensorboard', choices=[0, 1, 2], type=int,
                        help="If present, log to TensorBoard. Log levels:\n"
                             "0: Just log the loss, accuracy, and seq_len\n"
                             "1: Add histograms of biases and weights (Warning: slow)\n"
                             "2: Add histograms of biases and weights gradients (Warning: even slower)")
    parser.add_argument('--lf', dest='logging_frequency', default=100,  type=int,
                        help='TensorBoard logging frequency (Default: 100, i.e. logs every 100 episodes)')
    parser.add_argument('--log', action='store', dest='log', type=str, default='INFO',
                        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                        help="Log level. (Default: INFO)")
    parser.add_argument('--sma', dest='save_model_always', action='store_true', 
                        help='Stores model in every validation step, disregarding whether there was improvement or not (Default: False)')

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
    time_str = '{0:%Y%m%d_%H%M%S}'.format(datetime.now())
    log_dir = path_root + task_name + '/' + model_name + '/' + time_str + '/'
    os.makedirs(log_dir, exist_ok=False)
    log_file = log_dir + 'msgs.log'
    copyfile(FLAGS.task, log_dir + "/train_settings.yaml")  # Copy the task's yaml file into log_dir

    def logfile():
        return logging.FileHandler(log_file)

    with open('logger_config.yaml', 'rt') as f:
        config = yaml.load(f.read())
        logging.config.dictConfig(config)

    logger = logging.getLogger('Train')
    logger.setLevel(getattr(logging, FLAGS.log.upper(), None))

    # print experiment configuration
    str = 'Configuration for {}:\n'.format(task_name)
    str += yaml.safe_dump(config_loaded, default_flow_style=False,
                          explicit_start=True, explicit_end=True)
    logger.info(str)

    if FLAGS.confirm:
        # Ask for confirmation
        input('Press any key to continue')

    # set seed
    if config_loaded["settings"]["seed_torch"] != -1:
        torch.manual_seed(config_loaded["settings"]["seed_torch"])

    if config_loaded["settings"]["seed_numpy"] != -1:
        np.random.seed(config_loaded["settings"]["seed_numpy"])

    # Determine if CUDA is to be used
    use_CUDA = False
    if torch.cuda.is_available():
        try:  # If the 'cuda' key is not present, catch the exception and do nothing
            if config_loaded['problem_train']['cuda']:
                use_CUDA = True
        except KeyError:
            pass

    if config_loaded['problem_train']['curriculum_learning_interval'] != 0:
        config_loaded['problem_train']['max_sequence_length'] = config_loaded['problem_train']['min_sequence_length']

    # Build problem for the training
    problem = ProblemFactory.build_problem(config_loaded['problem_train'])

    # Build problem for the validation
    problem_validation = ProblemFactory.build_problem(config_loaded['problem_validation'])
    generator_validation = problem_validation.return_generator()
    
    # Get a single batch that will be used during all validation steps.
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

    # Create tensorboard output, if tensorboard chosen
    if FLAGS.tensorboard is not None:
        from tensorboardX import SummaryWriter
        training_writer = SummaryWriter(log_dir+'/training')
        validation_writer = SummaryWriter(log_dir+'/validation')

    # Start Training
    episode = 0
    best_loss = 0.2 # TK: WHY?
    last_losses = collections.deque()

    # Try to read validation frequency from config, else set default (100)
    try: 
        validation_frequency = config_loaded['problem_validation']['frequency']
    except KeyError:
        validation_frequency = 100
        pass

    train_file = open(log_dir + 'training.csv', 'w', 1)
    validation_file = open(log_dir + 'validation.csv', 'w', 1)
    train_file.write('episode,accuracy,loss,length\n')
    validation_file.write('episode,accuracy,length\n')

    # Data generator : input & target
    for inputs, targets, mask in problem.return_generator():
        # Convert inputs and targets to CUDA
        if use_CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # Curriculum learning stop condition.
        curric_done=True
        # apply curriculum learning
        try:  # If the 'curriculum_learning_interval' key is not present, catch the exception and do nothing
            if config_loaded['problem_train']['curriculum_learning_interval']  > 0:
                min_length=config_loaded['problem_train']['min_sequence_length']
                max_max_length=config_loaded['problem_train']['max_sequence_length']

                # the curriculum learning goes from the min length to the maximum max length in steps of size 1
                max_length = min_length + int(episode / config_loaded['problem_train']['curriculum_learning_interval'])
                if max_length > max_max_length:
                    max_length = max_max_length
                else:
                    curric_done=False
                problem.set_max_length(max_length)
        
        except KeyError:
            pass

        # reset gradients
        optimizer.zero_grad()

        # 1. Perform forward step, calculate logits, loss  and accuracy.
        logits,  loss,  accuracy = forward_step(model, DataTuple(inputs, targets, mask),  config_loaded['settings']['use_mask'],  criterion)
            
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
        if (FLAGS.tensorboard is not None) and (episode % FLAGS.logging_frequency== 0):
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


       # 5. Validation. check if new loss is smaller than the best loss, save the model in this case
        if loss < best_loss or ((episode+1) % validation_frequency) == 0:

            improved = False
            if loss < best_loss:
                best_loss = loss
                improved = True

            # Export model if better OR user simply wants to export the mode..
            if FLAGS.save_model_always or improved:
                torch.save(model.state_dict(), log_dir + "/model_parameters")
                logger.info("Model exported")

            # Calculate the accuracy and loss of the validation data
            length_valid = data_valid[0].size(-2)
            _,  loss_valid, accuracy_valid = forward_step(model, data_valid, config_loaded['settings']['use_mask'],  criterion)
            format_str = 'episode {:05d}; acc={:12.10f}; loss={:12.10f}; length={:d} [Validation]'
            if not improved:
                format_str = format_str + ' *'

            logger.info(format_str.format(episode, accuracy_valid, loss_valid, length_valid))
            format_str = '{:05d}, {:12.10f}, {:12.10f}, {:03d}'
            if not improved:
                format_str = format_str + ' *'

            format_str = format_str + '\n'
            validation_file.write(format_str.format(episode, accuracy_valid, loss_valid, length_valid))

            if (FLAGS.tensorboard is not None):
                # Save loss + accuracy to tensorboard
                validation_writer.add_scalar('Loss', loss_valid, episode)
                validation_writer.add_scalar('Accuracy', accuracy_valid, episode)
                validation_writer.add_scalar('Seq_len', length_valid, episode)
            # End of validation.
 

        if curric_done:
            # break if conditions applied: convergence or max episodes
            if max(last_losses) < config_loaded['settings']['loss_stop'] \
                or episode == config_loaded['settings']['max_episodes'] :
                break

        episode += 1

    train_file.close()
    validation_file.close()

    logger.info('Learning finished!')
