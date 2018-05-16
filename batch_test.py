"""
This scripts does a random search on DNC's hyper parameters.
It works by loading a template yaml file, modifying the resulting dict, and dumping that as yaml into a
temporary file. The `train.py` script is then launched using the temporary yaml file as the task.
It will run as many concurrent jobs as possible.
"""

import os
import sys
from multiprocessing.pool import ThreadPool
import subprocess
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Headless backend for matplotlib
import matplotlib.pyplot as plt


def main():
    batch_file = sys.argv[1]
    assert os.path.isdir(batch_file)

    # Load the list of yaml files to run
    with open(batch_file, 'r') as f:
        directory_checkpoints = [l.strip() for l in f.readlines()]
        for foldername in directory_checkpoints:
            assert os.path.isdir(foldername), foldername + " is not a file"

    experiments_list = []
    for elem in directory_checkpoints:
        list_path = os.walk(elem)
        _, subdir, _ = next(list_path)
        for sub in subdir:
            checkpoints = os.path.join(elem, sub)
            experiments_list.append(checkpoints)

    # Keep only the folders that contain validation.csv and training.csv
    experiments_list = [elem for elem in experiments_list
                        if os.path.isfile(elem + '/validation.csv') and os.path.isfile(elem + '/training.csv')]

    # Run in as many threads as there are CPUs available to the script
    with ThreadPool(processes=len(os.sched_getaffinity(0))) as pool:
        pool.map(run_experiment, experiments_list)


def run_experiment(path: str):
    val_episode, val_accuracy, val_loss, val_length = np.loadtxt(path + '/validation.csv', delimiter=', ', skiprows=1)
    train_episode, train_accuracy, train_loss, train_length = np.loadtxt(path + '/training.csv', delimiter=', ', skiprows=1)

    # Save plot of losses to png file
    plt.semilogy(val_episode, val_loss, label='validation loss')
    plt.semilogy(train_episode, train_loss, label='training loss')
    plt.savefig(path + '/loss.png')
    plt.close()

    command_str = "cuda-gpupick -n0 python3 test.py -i {0} ".format(path).split()

    with open(os.devnull, 'w') as devnull:
        result = subprocess.run(command_str, stdout=devnull)

    if result.returncode != 0:
        print("Training exited with code:", result.returncode)


if __name__ == '__main__':
    main()
