"""
This scripts does a random search on DNC's hyper parameters.
It works by loading a template yaml file, modifying the resulting dict, and dumping that as yaml into a
temporary file. The `train.py` script is then launched using the temporary yaml file as the task.
It will run as many concurrent jobs as possible.
"""

import os
import sys
import yaml
from random import randrange
from tempfile import NamedTemporaryFile
from multiprocessing.pool import ThreadPool
import subprocess


EXPERIMENT_REPETITIONS = 10


def main():
    batch_file = sys.argv[1]
    assert os.path.isfile(batch_file)

    # Load the list of yaml files to run
    with open(batch_file, 'r') as f:
        yaml_files = [l.strip() for l in f.readlines()]
        for filename in yaml_files:
            assert os.path.isfile(filename), filename + " is not a file"

    experiments_list = []
    for _ in range(EXPERIMENT_REPETITIONS):
        experiments_list.extend(yaml_files)

    # Run in as many threads as there are CPUs available to the script
    with ThreadPool(processes=len(os.sched_getaffinity(0))) as pool:
        pool.map(run_experiment, experiments_list)


def run_experiment(yaml_file_path: str):
    # Load template yaml file
    with open(yaml_file_path, 'r') as yaml_file:
        params = yaml.load(yaml_file)

    # Change some params to random ones with specified ranges
    params['settings']['loss_stop'] = 1.E-5
    params['settings']['max_episodes'] = 100000
    params['problem_train']['cuda'] = False
    params['problem_train']['min_sequence_length'] = 1
    params['problem_train']['max_sequence_length'] = 20
    params['problem_train']['curriculum_learning']['interval'] = 500
    params['problem_train']['curriculum_learning']['initial_max_sequence_length'] = 4
    params['problem_validation']['min_sequence_length'] = 21
    params['problem_validation']['max_sequence_length'] = 21
    params['settings']['seed_numpy'] = randrange(0, 2**32)
    params['settings']['seed_torch'] = randrange(0, 2**32)

    # Create temporary file, in which we dump the modified params dict as yaml
    with NamedTemporaryFile(mode='w') as temp_yaml:
        yaml.dump(params, temp_yaml, default_flow_style=False)

        command_str = "python3 train.py --c {0}".format(temp_yaml.name).split()

        with open(os.devnull, 'w') as devnull:
            result = subprocess.run(command_str, stdout=devnull)

        if result.returncode != 0:
            print("Training exited with code:", result.returncode)


if __name__ == '__main__':
    main()
