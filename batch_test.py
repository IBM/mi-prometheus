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
from itertools import repeat
from tempfile import NamedTemporaryFile
from multiprocessing.pool import ThreadPool
import subprocess

def main():
    batch_file = sys.argv[1]
    assert os.path.isdir(batch_file)

    # Load the list of yaml files to run
    with open(batch_file, 'r') as f:
        directory_checkpoints = [l.strip() for l in f.readlines()]
        for folderename in directory_checkpoints:
            assert os.path.isdir(folderename), folderename + " is not a file"

    experiments_list = []
    for elem in directory_checkpoints:
        list_path = os.walk(elem)
        _, subdir, _ = next(list_path)
        for sub in subdir:
            checkpoints = os.path.join(elem, sub)
            experiments_list.append(checkpoints)

    # Run in as many threads as there are CPUs available to the script
    with ThreadPool(processes=len(os.sched_getaffinity(0))) as pool:
        pool.map(run_experiment, experiments_list)


def run_experiment(path: str):

        command_str = "cuda-gpupick -n0 python3 test.py -i {0} ".format(path).split()

        with open(os.devnull, 'w') as devnull:
            result = subprocess.run(command_str, stdout=devnull)

        if result.returncode != 0:
            print("Training exited with code:", result.returncode)


if __name__ == '__main__':
    main()
