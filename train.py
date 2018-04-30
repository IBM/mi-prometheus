# Force MKL (CPU BLAS) to use one core, faster
import os
os.environ["OMP_NUM_THREADS"] = '1'

import yaml
import os.path
import argparse
import torch
from torch import nn
import collections
import numpy as np

# Import problems and problem factory.
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'problems'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from problems.problem_factory import ProblemFactory
from models.model_factory import ModelFactory

if __name__ == '__main__':

    # Create parser with list of  runtime arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default='', dest='task',
                        help='Name of the task configuration file to be loaded')
    parser.add_argument('-m', action='store_true', dest='mode',
                        help='Mode (TRUE: trains a new model, FALSE: tests existing model)')
    parser.add_argument('-i', type=int, default='100000', dest='iterations', help='Number of training epochs')
    # Parse arguments.
    FLAGS, unparsed = parser.parse_known_args()

    # Check if config file was selected.
    if (FLAGS.task == ''):
        print('Please pass task configuration file as -t parameter')
        exit(-1)
    # Check it file exists.
    if not os.path.isfile(FLAGS.task):
        print('Task configuration file {} does not exists'.format(FLAGS.task))
        exit(-2)

    # Read YAML file
    with open(FLAGS.task, 'r') as stream:
        config_loaded = yaml.load(stream)

    # set seed
    if config_loaded["settings"]["seed"] != -1:
        torch.manual_seed(2)
        np.random.seed(config_loaded["settings"]["seed"])

    # Print loaded configuration
    # print("Loaded configuration",  config_loaded)
    print("Problem configuration:\n", config_loaded['problem_train'])
    print("Model configuration:\n", config_loaded['model'])
    print("settings configuration:\n", config_loaded['settings'])

    # Determine if CUDA is to be used
    use_CUDA = False
    if torch.cuda.is_available():
        try:  # If the 'cuda' key is not present, catch the exception and do nothing
            if config_loaded['problem_train']['cuda']:
                use_CUDA = True
        except KeyError:
            None

    # Build problem
    problem = ProblemFactory.build_problem(config_loaded['problem_train'])

    # Build model
    model = ModelFactory.build_model(config_loaded['model'])
    model.cuda() if use_CUDA else None

    # Set loss and optimizer
    optimizer_conf = dict(config_loaded['optimizer'])
    optimizer_name = optimizer_conf['name']
    del optimizer_conf['name']

    criterion = nn.BCELoss()
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_conf)

    # Start Training
    epoch = 0
    last_losses = collections.deque()

    # Data generator : input & target
    for inputs, targets, mask in problem.return_generator():
        # Convert inputs and targets to CUDA
        if use_CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()

        # apply model
        output = model(inputs)

        # compute loss
        # TODO: solution for now - mask[0]
        if config_loaded['settings']['use_mask']:
            loss = criterion(output[:, mask[0], :], targets[:, mask[0], :])
        else:
            loss = criterion(output, targets)

        # print statistics
        print("epoch: {:5d}, loss: {:1.6f}, length: {:02d}".format(epoch + 1, loss, inputs.size(-2)))

        # append the new loss
        last_losses.append(loss)
        if len(last_losses) > config_loaded['settings']['length_loss']:
            last_losses.popleft()

        loss.backward()

        # clip grad between -10, 10
        nn.utils.clip_grad_value_(model.parameters(), 10)

        optimizer.step()

        if max(last_losses) < config_loaded['settings']['loss_stop'] \
                or epoch == config_loaded['settings']['max_epochs']:
            path = "./checkpoints/"
            # save model parameters
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model.state_dict(), path+"model_parameters"+ '_' +config_loaded['problem_train']['name'])
            break

        epoch += 1

    print("Learning finished!")

