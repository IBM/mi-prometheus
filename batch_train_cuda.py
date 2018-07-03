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
from time import sleep

EXPERIMENT_REPETITIONS = 10
MAX_THREADS = 6 

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
#    with ThreadPool(processes=len(os.sched_getaffinity(0))) as pool:
 #       pool.map(run_experiment, experiments_list)
    with ThreadPool(processes=MAX_THREADS) as pool:
        thread_results = []  # This contains a list of `AsyncResult` objects. To check if completed and get result.

        for task in experiments_list:
            thread_results.append(pool.apply_async(run_experiment, (task,)))
            print("Started training", task)

            # Check every 3 seconds if there is a (supposedly) free GPU to start a task on
            sleep(3)
            while [r.ready() for r in thread_results].count(False) >= MAX_THREADS:
                sleep(3)

        # Equivalent of what would usually be called "join" for threads
        for r in thread_results:
            r.wait()

def run_experiment(yaml_file_path: str):
    # Load template yaml file
    with open(yaml_file_path, 'r') as yaml_file:
        params = yaml.load(yaml_file)

    # Change some params to random ones with specified ranges
    params['settings']['loss_stop'] = 1.E-5
    params['settings']['max_episodes'] = 100000
    params['problem_train']['cuda'] = True
    params['problem_test']['cuda'] = True
    params['problem_train']['control_bits'] = 3
    params['problem_validation']['control_bits'] = 3
    params['problem_test']['control_bits'] = 3
    params['problem_train']['min_sequence_length'] = 3
    params['problem_train']['max_sequence_length'] = 20
    params['problem_train']['curriculum_learning']['interval'] = 500
    params['problem_train']['curriculum_learning']['initial_max_sequence_length'] = 5
    params['problem_validation']['min_sequence_length'] = 21
    params['problem_validation']['max_sequence_length'] = 21
    params['problem_test']['min_sequence_length'] = 1000
    params['problem_test']['max_sequence_length'] = 1000
    
    params['problem_validation']['frequency'] = 1000
    params['model']['num_layers'] = 3
    params['model']['hidden_state_dim'] = 512
    try:
        params['model']['memory']['num_content_bits'] = 15
    except KeyError:
        pass
    try:
        params['model']['memory_content_size'] = 15
    except KeyError:
        pass

    try:
        params['model']['num_control_bits'] = 3
    except KeyError:
        pass
    try:
        params['model']['control_bits'] = 3
    except KeyError:
        pass


    params['settings']['seed_numpy'] = randrange(0, 2**32)
    params['settings']['seed_torch'] = randrange(0, 2**32)

    # Create temporary file, in which we dump the modified params dict as yaml
    with NamedTemporaryFile(mode='w') as temp_yaml:
        yaml.dump(params, temp_yaml, default_flow_style=False)

        command_str = "cuda-gpupick -n1 python3 train.py --c {0}".format(temp_yaml.name).split()

        with open(os.devnull, 'w') as devnull:
            result = subprocess.run(command_str, stdout=devnull)

        if result.returncode != 0:
            print("Training exited with code:", result.returncode)


if __name__ == '__main__':
    main()
