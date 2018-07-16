"""
This script runs test.py on the output of batch_train.
The input is a list of directories for each problem/model e.g. experiments/serial_recall/dnc
 and executes on every run of the model in that directory. I.e. if you tell it to run on serial_recall/dnc, 
it will process every time you have ever run serial_recall with the DNC. This should be fixed later.
"""


import os
import sys
import yaml
from multiprocessing.pool import ThreadPool
import subprocess
import numpy as np
from glob import glob
import csv
import pandas as pd
from time import sleep

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

MAX_THREADS = 6

def main():
    batch_file = sys.argv[1]
    assert os.path.isfile(batch_file)

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
    # check if the files are empty except for the first line
    experiments_list = [elem for elem in experiments_list
                        if os.stat(elem + '/validation.csv').st_size > 24  and os.stat(elem + '/training.csv').st_size > 24 ]


    # Run in as many threads as there are CPUs available to the script
    with ThreadPool(processes=MAX_THREADS) as pool:
        thread_results = []  # This contains a list of `AsyncResult` objects. To check if completed and get result.

        for task in experiments_list:
            thread_results.append(pool.apply_async(run_experiment, (task,)))
            print("Started testing", task)

            # Check every 3 seconds if there is a (supposedly) free GPU to start a task on
            sleep(3) 
            while [r.ready() for r in thread_results].count(False) >= MAX_THREADS:
                sleep(3) 
    
        # Equivalent of what would usually be called "join" for threads
        for r in thread_results:
            r.wait()

def run_experiment(path: str):

    # Load yaml file. To get model name and problem name.
    with open(path + '/train_settings.yaml', 'r') as yaml_file:
        params = yaml.load(yaml_file)

    # print path
    print(path)

    valid_csv = pd.read_csv(path + '/validation.csv', delimiter=',', header=0)
    train_csv = pd.read_csv(path + '/training.csv', delimiter=',', header=0)

    # best train point
    index_val_loss = pd.Series.idxmin(train_csv.loss)  
    train_episodes = train_csv.episode.values.astype(int)  # best train loss argument
    best_train_ep = train_episodes[index_val_loss]  # best train loss argument
    best_train_loss = train_csv.loss[index_val_loss]
    
    # best valid point 
    index_val_loss = pd.Series.idxmin(valid_csv.loss) 
    valid_episodes = valid_csv.episode.values.astype(int)  # best train loss argument
    best_valid_ep = valid_episodes[index_val_loss]  # best train loss argument
    best_valid_loss = valid_csv.loss[index_val_loss]

         
    ### Find the best model ###
    models_list3 = glob(path + '/models/model_episode*')
    models_list2 = [os.path.basename(os.path.normpath(e)) for e in models_list3]
    models_list = [int(e.split('_')[-1].split('.')[0]) for e in models_list2]

    # check if models list is empty
    if models_list:
        # select the best model 
        best_num_model, idx_best = find_nearest(models_list, best_valid_ep)
        
        last_model, idx_last = find_nearest(models_list, valid_episodes[-1]) 
             
        # Run the test
        command_str = "cuda-gpupick -n1 python3 test.py --model {0}".format(models_list3[idx_best]).split()
        with open(os.devnull, 'w') as devnull:
            result = subprocess.run(command_str, stdout=devnull)
        if result.returncode != 0:
            print("Testing exited with code:", result.returncode)
   
    
    else:
        print('There is no model in checkpoint {} '.format(path))     


if __name__ == '__main__':
    main()
