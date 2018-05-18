"""
This scripts does a random search on DNC's hyper parameters.
It works by loading a template yaml file, modifying the resulting dict, and dumping that as yaml into a
temporary file. The `train.py` script is then launched using the temporary yaml file as the task.
It will run as many concurrent jobs as possible.
"""

import os
import sys
import yaml
from multiprocessing.pool import ThreadPool
import subprocess
import numpy as np
from glob import glob
import csv

import matplotlib
matplotlib.use('Agg')  # Headless backend for matplotlib
import matplotlib.pyplot as plt

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

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

    # Run in as many threads as there are CPUs available to the script
    with ThreadPool(processes=len(os.sched_getaffinity(0))) as pool:
        list_dict_exp = pool.map(run_experiment, experiments_list)
        exp_values = dict(zip(list_dict_exp[0], zip(*[d.values() for d in list_dict_exp])))
         
        with open(directory_checkpoints[0].split("/")[0] + "_test.csv", "w") as outfile:
            writer = csv.writer(outfile, delimiter = " ")
            writer.writerow(exp_values.keys())
            writer.writerows(zip(*exp_values.values()))
          

def run_experiment(path: str):
    r = {}  # results dictionary
      
    r['timestamp'] = os.path.basename(os.path.normpath(path))

    # Load yaml file. To get model name and problem name.
    with open(path + '/train_settings.yaml', 'r') as yaml_file:
        params = yaml.load(yaml_file)
    r['model'] = params['model']['name']
    r['problem'] = params['problem_train']['name']

    # print path
    print(path)

    # Load csv files
    val_episode, val_accuracy, val_loss, val_length = \
        np.loadtxt(path + '/validation.csv', delimiter=', ', skiprows=1, unpack=True, dtype='int, float, float, int') 
    train_episode, train_accuracy, train_loss, train_length = \
        np.loadtxt(path + '/training.csv', delimiter=', ', skiprows=1, unpack=True, dtype='int, float, float, int')

    # Save plot of losses to png file
    # Save plot of losses to png file
    try:
        ax = plt.gca()
        ax.semilogy(val_episode, val_loss, label='validation loss')
        ax.semilogy(train_episode, train_loss, label='training loss')
        plt.savefig(path + '/loss.png')
        plt.close()
    except:
       pass 
    ### ANALYSIS OF TRAINING AND VALIDATION DATA ###

    index_val_loss = np.argmin(val_loss) 
    r['best_valid_arg'] = int(val_episode[index_val_loss])  # best validation loss argument
    r['best_valid_loss'] = val_loss[index_val_loss]
    r['best_valid_accuracy'] = val_accuracy[index_val_loss]   
 
    # If the best loss < .1, keep that as the early stopping point
    # Otherwise, we take the very last data as the stopping point
    if val_loss[index_val_loss] < 1.E-4:
        r['converge'] = True
    else:
        r['converge'] = False     
 
    r['stop_episode'] = train_episode[-1]
    stop_train_index = -1 
    index_val_loss = -1
     
    ### Find the best model ###
    models_list = glob(path + '/models/*')
    models_list = [os.path.basename(os.path.normpath(e)) for e in models_list]
    models_list = [int(e.split('_')[-1]) for e in models_list]    
   
    # Gather data at chosen stopping point
    #r['valid_loss'] = val_loss[index_val_loss]
    #r['valid_accuracy'] = val_accuracy[index_val_loss]
    #r['valid_length'] = val_length[index_val_loss]

    # check if models list is empty
    if models_list:
        # select the best model 
        best_num_model = find_nearest(models_list, r['best_valid_arg'])
        
        last_model = find_nearest(models_list, train_episode[-1])
      
        # to avoid selecting model zeros, if training is not converging 
        if best_num_model == 0:
            best_num_model = 1000 # hack for now 
        
        r['best_model'] = best_num_model  
        print(best_num_model)        
        # Run the test
        command_str = "cuda-gpupick -n0 python3 test.py -i {0} -e {1} -f {2}".format(path, best_num_model,last_model).split()
        with open(os.devnull, 'w') as devnull:
            result = subprocess.run(command_str, stdout=devnull)
        if result.returncode != 0:
            print("Testing exited with code:", result.returncode)
   
        # Load the test results from csv
        test_episode, test_accuracy, test_loss, test_length = \
            np.loadtxt(path + '/test.csv', delimiter=', ', skiprows=1, unpack=True)

        # Load the test results from csv
        test_train_episode, test_train_accuracy, test_train_loss, test_train_length = \
            np.loadtxt(path + '/test_train.csv', delimiter=', ', skiprows=1, unpack=True)
    
        # Save test results into dict. We expect that the csv has a single row of data
        r['train_loss'] = test_train_loss
        r['train_accuracy'] = test_train_accuracy
        r['train_length'] = test_train_length
        r['test_loss'] = test_loss
        r['test_accuracy'] = test_accuracy
        r['test_length'] = test_length
    else:
        print('There is no model in checkpoint {} '.format(path))     

    return r


if __name__ == '__main__':
    main()
