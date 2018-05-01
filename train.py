# Force MKL (CPU BLAS) to use one core, faster
import os
os.environ["OMP_NUM_THREADS"] = '1'

import yaml
import os.path
import argparse
import torch
from torch import nn
import torch.nn.functional as F
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
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-t', type=str, default='', dest='task',
                        help='Name of the task configuration file to be loaded')
    parser.add_argument('-m', action='store_true', dest='mode',
                        help='Mode (TRUE: trains a new model, FALSE: tests existing model)')
    parser.add_argument('-i', type=int, default='100000', dest='iterations', help='Number of training epochs')
    parser.add_argument('--tensorboard', action='store', dest='tensorboard', choices=[0, 1, 2], type=int,
                        help="If present, log to tensorboard. Log levels:\n"
                             "0: Just log the loss, accuracy, and seq_len\n"
                             "1: Add histograms of biases and weights (Warning: slow)\n"
                             "2: Add histograms of biases and weights gradients (Warning: even slower)")
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

    # Prepare output paths
    path_root = "./checkpoints/"
    path_out = path_root + config_loaded['problem_train']['name']
    os.makedirs(path_out, exist_ok=True)

    # set seed
    if config_loaded["settings"]["seed_torch"] != -1:
        torch.manual_seed(config_loaded["settings"]["seed_torch"])

    if config_loaded["settings"]["seed_numpy"] != -1:
        np.random.seed(config_loaded["settings"]["seed_numpy"])

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

    criterion = nn.BCEWithLogitsLoss()
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_conf)

    # Create tensorboard output, if tensorboard chosen
    if FLAGS.tensorboard is not None:
        from tensorboardX import SummaryWriter
        tb_writer = SummaryWriter(path_out)

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

        # append the new loss
        last_losses.append(loss)
        if len(last_losses) > config_loaded['settings']['length_loss']:
            last_losses.popleft()

        loss.backward()

        # clip grad between -10, 10
        nn.utils.clip_grad_value_(model.parameters(), 10)

        optimizer.step()

        # print statistics
        print("epoch: {:5d}, loss: {:1.6f}, length: {:02d}".format(epoch + 1, loss, inputs.size(-2)))

        if FLAGS.tensorboard is not None:
            # Save loss + accuracy to tensorboard
            accuracy = (1 - torch.abs(torch.round(F.sigmoid(output)) - targets)).mean()
            tb_writer.add_scalar('Train/loss', loss, epoch)
            tb_writer.add_scalar('Train/accuracy', accuracy, epoch)
            tb_writer.add_scalar('Train/seq_len', inputs.size(-2), epoch)

            for name, param in model.named_parameters():
                if FLAGS.tensorboard >= 1:
                    tb_writer.add_histogram(name, param.data.cpu().numpy(), epoch)
                if FLAGS.tensorboard >= 2:
                    tb_writer.add_histogram(name + '/grad', param.grad.data.cpu().numpy(), epoch)

        if max(last_losses) < config_loaded['settings']['loss_stop'] \
                or epoch == config_loaded['settings']['max_epochs']:
            # save model parameters
            torch.save(model.state_dict(), path_out + "/model_parameters")
            tb_writer.export_scalars_to_json(path_out + "/all_scalars.json")
            tb_writer.close()
            break

        epoch += 1

    print("Learning finished!")

