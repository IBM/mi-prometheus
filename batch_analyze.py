"""
This script post processes the output of batch_train and batch_test. It takes as input the same file as batch_test and executes on every run of the model in that directory. I.e. if you tell it to run on serial_recall/dnc, it will process every time you have ever run serial_recall with the DNC as long as test.py has been executed. This should be fixed later.
"""

import os
import sys
import yaml
from multiprocessing.pool import ThreadPool
import numpy as np
from glob import glob
import csv
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for matplotlib
import matplotlib.pyplot as plt

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

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
    experiments_list = [elem for elem in experiments_list
                        if os.path.isfile(elem + '/test.csv')]

    # check if the files are empty except for the first line
    experiments_list = [elem for elem in experiments_list
                        if os.stat(elem + '/validation.csv').st_size > 24  and os.stat(elem + '/training.csv').st_size > 24 ]
    experiments_list = [elem for elem in experiments_list
                        if os.stat(elem + '/test.csv').st_size > 24]



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
    run_test = True  
      
    r['timestamp'] = os.path.basename(os.path.normpath(path))

    # Load yaml file. To get model name and problem name.
    with open(path + '/train_settings.yaml', 'r') as yaml_file:
        params = yaml.load(yaml_file)
    r['model'] = params['model']['name']
    r['problem'] = params['problem_train']['name']

    # print path
    print(path)

    
    valid_csv = pd.read_csv(path + '/validation.csv', delimiter=',', header=0)
    test_csv = pd.read_csv(path + '/test.csv', delimiter=',', header=0)
    train_csv = pd.read_csv(path + '/training.csv', delimiter=',', header=0)

    # best train point
    train_episode = train_csv.episode.values.astype(int)  # best train loss argument
    train_loss = train_csv.loss.values.astype(float)  # best train loss argument
    if 'acc' in train_csv:
       train_accuracy = train_csv.acc.values.astype(float) 
 

    index_train_loss = np.argmin(train_loss)
    best_train_ep = train_episode[index_train_loss]  # best train loss argument
    best_train_loss = train_loss[index_train_loss]

    # best valid point 
    valid_episode = valid_csv.episode.values.astype(int)  # best train loss argument
    valid_loss = valid_csv.loss.values.astype(float)  # best train loss argument
    if 'acc' in valid_csv:
       valid_accuracy = valid_csv.acc.values.astype(float) 
   

 
    index_val_loss = np.argmin(valid_loss)
    best_valid_ep = valid_episode[index_val_loss]  # best train loss argument
    best_valid_loss = valid_loss[index_val_loss]


    # Save plot of losses to png file
    # Save plot of losses to png file
    try:
        ax = plt.gca()
        ax.semilogy(valid_episode, valid_loss, label='validation loss')
        ax.semilogy(train_episode, train_loss, label='training loss')
        plt.savefig(path + '/loss.png')
        plt.close()
    except:
       pass 
    ### ANALYSIS OF TRAINING AND VALIDATION DATA ###

    # best valid train 
    index_val_loss = np.argmin(valid_loss) 
    r['best_valid_arg'] = int(valid_episode[index_val_loss])  # best validation loss argument
    r['best_valid_loss'] = valid_loss[index_val_loss]
    
    if 'acc' in valid_csv:
        r['best_valid_accuracy'] = valid_accuracy[index_val_loss]   
 
    # best train loss
    index_loss = np.where(train_loss<1.E-4)[0]
        
    if index_loss.size: 
        r['best_train_arg'] = int(train_episode[index_loss[0]])  # best validation loss argument
        r['best_train_loss'] = train_loss[index_loss[0]]

        if 'acc' in train_csv:
            r['best_train_accuracy'] = train_accuracy[index_loss[0]]
    else: 
        index_loss = np.argmin(train_loss)
        r['best_train_arg'] = int(train_episode[index_loss])  # best validation loss argument
        r['best_train_loss'] = train_loss[index_loss]
        if 'acc' in train_csv:
            r['best_train_accuracy'] = train_accuracy[index_loss]

    # If the best loss < .1, keep that as the early stopping point
    # Otherwise, we take the very last data as the stopping point
    if valid_loss[index_val_loss] < 1.E-4:
        r['converge'] = True
    else:
        r['converge'] = False     
 
    r['stop_episode'] = train_episode[-1]
    stop_train_index = -1 
    index_val_loss = -1
     
    ### Find the best model ###
    models_list3 = glob(path + '/models/model_episode_*')
    models_list2 = [os.path.basename(os.path.normpath(e)) for e in models_list3]
    models_list = [int(e.split('_')[-1].split('.')[0]) for e in models_list2]    
   
    # Gather data at chosen stopping point
    #r['valid_loss'] = val_loss[index_val_loss]
    #r['valid_accuracy'] = val_accuracy[index_val_loss]
    #r['valid_length'] = val_length[index_val_loss]

    # check if models list is empty
    if models_list and run_test:
        # select the best model 
        best_num_model, idx_best = find_nearest(models_list, r['best_valid_arg'])
        
        last_model, idx_last = find_nearest(models_list, train_episode[-1])
      
        # to avoid selecting model zeros, if training is not converging 
        if best_num_model == 0:
            best_num_model = 1000 # hack for now 
        
        r['best_model'] = best_num_model  
                
        # best test point 
        r['test_loss'] = test_csv.loss.values.astype(float)  # best train loss argument
        if 'acc' in valid_csv:
            r['test_accuracy']= test_csv.acc.values.astype(float) 

    else:
        print('There is no model in checkpoint {} '.format(path))     

    return r


if __name__ == '__main__':
    main()
