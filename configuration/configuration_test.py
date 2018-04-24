#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""configuration_test.py: Test of YAML configuration"""
__author__      = "Tomasz Kornuta"

import yaml
import os.path
import argparse

if __name__ == '__main__':
 
    # Create parser with list of  runtime arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default='',  dest='task', help='Name of the task configuration file to be loaded')
    parser.add_argument('-m', action='store_true', dest='mode',  help='Mode (TRUE: trains a new model, FALSE: tests existing model)')
    parser.add_argument('-i', type=int, default='100000',  dest='iterations', help='Number of training epochs')
    # Parse arguments.
    FLAGS, unparsed = parser.parse_known_args()

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

    # Print loaded configuration
    #print("Loaded configuration",  config_loaded)    
    print("Problem configuration:\n",  config_loaded['problem'])
    print("Model configuration:\n",  config_loaded['model'])
    
    # Run mode: training or inference.
    #if FLAGS.mode:
    #    run_training(FLAGS.iterations,  FLAGS.checkpoint)
    #else:
    #    run_inference(FLAGS.checkpoint)
