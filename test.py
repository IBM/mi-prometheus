# Force MKL (CPU BLAS) to use one core, faster
import os
os.environ["OMP_NUM_THREADS"] = '1'

import torch
import numpy as np
import argparse
import yaml

# Force MKL (CPU BLAS) to use one core, faster
os.environ["OMP_NUM_THREADS"] = '1'

# Import problems and problem factory.
from problems.problem_factory import ProblemFactory
# Import models and model factory.
from models.model_factory import ModelFactory

if __name__ == '__main__':
    # set random seed
    np.random.seed(999999999)

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
    problem = ProblemFactory.build_problem(config_loaded['problem'])

    # Build model
    model = ModelFactory.build_model(config_loaded['model'])

    # load the trained model
    model.load_state_dict(torch.load(path+"model_parameters"))

    for inputs, targets, mask in problem.return_generator_random_length():
        # apply the trained model
        output = model(inputs)

        # test accuracy
        output = torch.round(output[:, mask, :])
        acc = 1 - torch.abs(output-targets)
        accuracy = acc.mean()
        print("Accuracy: %.6f" % (accuracy * 100) + "%")

        break   # one test sample




