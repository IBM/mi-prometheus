# Force MKL (CPU BLAS) to use one core, faster
import os
os.environ["OMP_NUM_THREADS"] = '1'

import torch
import argparse
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Force MKL (CPU BLAS) to use one core, faster
os.environ["OMP_NUM_THREADS"] = '1'

# Import problems and problem factory.
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'problems'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from problems.problem_factory import ProblemFactory
from models.model_factory import ModelFactory


def show_sample(inputs, targets, mask, sample_number=0):
    """ Shows the sample (both input and target sequences) using matplotlib."""
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]}, sharex=True)
    # Set ticks.
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Set labels.
    ax1.set_title('Output')
    ax1.set_ylabel('Control/Data bits')
    ax1.set_xlabel('Item number')
    ax2.set_title('Target')
    # ax2.set_ylabel('Data bits')
    ax2.set_xlabel('Item number')

    # Set data.
    ax1.imshow(np.transpose((inputs[sample_number, :, :]).detach().numpy(), [1, 0]))
    ax2.imshow(np.transpose((inputs[sample_number, :, :]).detach().numpy(), [1, 0]))

    plt.show()

    # Plot!

if __name__ == '__main__':
    # set random seed
    np.random.seed(9)

    # Create parser with list of  runtime arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default='', dest='task',
                        help='Name of the task configuration file to be loaded')
    parser.add_argument('-m', action='store_true', dest='mode',
                        help='Mode (TRUE: trains a new model, FALSE: tests existing model)')
    # Parse arguments.
    FLAGS, unparsed = parser.parse_known_args()

    # read training arguments
    path = "./checkpoints/"

    # Test
    print("Testing")

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

    # Build new problem
    problem = ProblemFactory.build_problem(config_loaded['problem_test'])

    # Build model
    model = ModelFactory.build_model(config_loaded['model'])

    # load the trained model
    model.load_state_dict(torch.load(path+"model_parameters" + '_' +config_loaded['problem_test']['name']))

    for inputs, targets, mask in problem.return_generator():
        # apply the trained model
        output = model(inputs)

        if config_loaded['settings']['use_mask']:
            output = output[:, mask[0], :]
            targets = targets[:, mask[0], :]

        # test accuracy
        output = torch.round(output)
        acc = 1 - torch.abs(output-targets)
        accuracy = acc.mean()
        print("Accuracy: %.6f" % (accuracy * 100) + "%")
        # plot data
        show_sample(output, targets, mask)

        break   # one test sample




