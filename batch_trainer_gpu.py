"""
This scripts does a random search on DNC's hyper parameters.
It works by loading a template yaml file, modifying the resulting dict, and dumping that as yaml into a
temporary file. The `train.py` script is then launched using the temporary yaml file as the task.
It will run as many concurrent jobs as possible.
"""

__author__= "Alexis Asseman, Younes Bouhadjar"

import os
import sys
import yaml
from tempfile import NamedTemporaryFile
from multiprocessing.pool import ThreadPool
import subprocess
from time import sleep
import argparse

def main():
    # Create parser with list of  runtime arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config', dest='config', type=str, default='',
                        help='Name of the batch configuration file to be loaded')

    # Parse arguments.
    FLAGS, unparsed = parser.parse_known_args()

    # Check if config file was selected.
    if FLAGS.config == '':
        print('Please pass batch configuration file as --c parameter')
        exit(-1)

    # Check if file exists.
    if not os.path.isfile(FLAGS.config):
        print('Error: Batch configuration file {} does not exist'.format(FLAGS.config))
        # raise Exception('Error: Configuration file {} does not exist'.format(config))
        exit(-1)

    try:
        # Open file and get parameter dictionary.
        with open(FLAGS.config, 'r') as stream:
            batch_dict = yaml.safe_load(stream)
    except yaml.YAMLError:
        print("Error: Coudn't properly parse the {} batch configuration file".format(FLAGS.config))
        exit(-1)

    # Get batch settings.
    try:
        experiment_repetitions = batch_dict['batch_settings']['experiment_repetitions']
        max_concurrent_runs = batch_dict['batch_settings']['max_concurrent_runs']
    except:
        print("Error: The 'batch_settings' section must define 'experiment_repetitions' and 'max_concurrent_runs'")
        exit(-1)

    # Check the presence of batch_overwrite section.
    if 'batch_overwrite' not in batch_dict:
        batch_overwrite_filename = None
    else:
        # Create temporary file with settings that will be overwritten for all tasks.
        batch_overwrite_file = NamedTemporaryFile(mode='w')
        yaml.dump(batch_dict['batch_overwrite'], batch_overwrite_file, default_flow_style=False)
        batch_overwrite_filename = batch_overwrite_file.name

    # Check the presence of tasks section.
    if 'batch_tasks' not in batch_dict:
        print("Error: Batch configuration is lacking the 'batch_tasks' section")
        exit(-1)

    # Create a configuration specific to this batch trainer: set seeds to random and cuda to false.
    gpu_batch_trainer_default_params = {"training": {"seed_numpy": -1, "seed_torch": -1, "cuda": True}}
    # Create temporary file
    gpu_batch_trainer_default_params_file = NamedTemporaryFile(mode='w')
    yaml.dump(gpu_batch_trainer_default_params, gpu_batch_trainer_default_params_file, default_flow_style=False)
    
        
 
    configs = []
    # Iterate through batch tasks.
    for task in batch_dict['batch_tasks']:
        try:
            # Retrieve the config(s).
            current_configs = gpu_batch_trainer_default_params_file.name + ',' + task['default_configs']
            # Extend them by batch_overwrite.
            if batch_overwrite_filename is not None:
                current_configs = batch_overwrite_filename + ',' + current_configs
            if 'overwrite' in task:
                # Create temporary file with settings that will be overwritten only for that particular task.
                overwrite_file = NamedTemporaryFile(mode='w')
                yaml.dump(task['overwrite'], overwrite_file, default_flow_style=False)
                current_configs = overwrite_file.name + ',' + current_configs

            # Get list of configs that need to be loaded.
            configs.append(current_configs)
            print(current_configs)
        except KeyError:
            pass

    # Create list of experiments by
    experiments_list = []
    for _ in range(experiment_repetitions):
        experiments_list.extend(configs)

    # Run in as many threads as there are GPUs available to the script
    # with ThreadPool(processes=len(os.sched_getaffinity(0))) as pool:
    # pool.map(run_experiment, experiments_list)
    with ThreadPool(processes=max_concurrent_runs) as pool:
        thread_results = []  # This contains a list of `AsyncResult` objects. To check if completed and get result.

        for task in experiments_list:
            thread_results.append(pool.apply_async(run_experiment, (task,)))
            print("Started training", task)

            # Check every 3 seconds if there is a (supposedly) free GPU to start a task on
            sleep(3)
            while [r.ready() for r in thread_results].count(False) >= max_concurrent_runs:
                sleep(3)

        # Equivalent of what would usually be called "join" for threads
        for r in thread_results:
            r.wait()


def run_experiment(experiment_configs: str):
    """ Runs the experiment.

    :param experiment_configs: List of configs (separated with coma) that will be passed to trainer.
    """

    command_str = "cuda-gpupick -n1 python3 trainer.py --c {0}".format(experiment_configs)

    print("Starting: ", command_str)
    with open(os.devnull, 'w') as devnull:
        result = subprocess.run(command_str, shell=True, stdout=devnull)
    print("Finished: ", command_str)

    if result.returncode != 0:
        print("Training exited with code:", result.returncode)


if __name__ == '__main__':
    main()
