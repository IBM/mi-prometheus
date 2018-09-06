#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""worker_utils.py: Contains helper functions for different workers"""
__author__      = "Ryan McAvoy, Tomasz Kornuta"

import os
import yaml
import numpy as np
import torch
from misc.app_state import AppState
from torch.nn.modules.module import _addindent

def forward_step(model, problem, episode, stat_col, data_tuple,  aux_tuple):
    """ Function performs a single forward step.

    :returns: logits, loss and accuracy (former using provided criterion)
    """
    # convert to CUDA
    if AppState().use_CUDA:
        data_tuple, aux_tuple = problem.turn_on_cuda(data_tuple, aux_tuple)

    # Perform forward calculation.
    logits = model(data_tuple)

    # Evaluate loss function.
    loss = problem.evaluate_loss(data_tuple, logits, aux_tuple)

    # Collect "elementary" statistics - episode and loss.
    stat_col['episode'] = episode
    stat_col['loss'] = loss

    # Collect other (potential) statistics from problem & model.
    problem.collect_statistics(stat_col, data_tuple, logits, aux_tuple)
    model.collect_statistics(stat_col, data_tuple, logits)

    # Return tuple: logits, loss.
    return logits, loss 

def check_and_set_cuda(params, logger):
    """ Enables Cuda if available and sets the default data types
    :param params: paramater interface object containing either training or testing parameters
    :param logger: logger object
    """
    turn_on_cuda = False
    try:  # If the 'cuda' key is not present, catch the exception and do nothing
        turn_on_cuda = params['cuda']
    except KeyError:
        pass

    # Determine if CUDA is to be used.
    if torch.cuda.is_available():
        if turn_on_cuda:
            AppState().convert_cuda_types()
            logger.info('Running with CUDA enabled')
    elif turn_on_cuda:
        logger.warning('CUDA is enabled but there is no available device')

    #TODO Add flags to change these
    AppState().set_dtype('float')
    AppState().set_itype('int')


def recurrent_config_parse(configs, configs_parsed):
    """ Function parses names of configuration files in a recursive mannner, i.e. 
    by looking for 'default_config' sections and trying to load and parse those files one by one.

    :param configs: String containing names of configuration files (with paths), separated by comas.
    :param configs_parsed: List of configurations that were already parsed (so we won't parse them many times).
    :returns: list of parsed configuration files.
    """
    # Split and remove spaces.
    configs_to_parse = configs.replace(" ", "").split(',')
    #configs_to_parse = ''.join(configs.split()).split(',')

    # Terminal condition.
    while len(configs_to_parse) > 0:

        # Get config.
        config = configs_to_parse.pop(0)
        # Skip empty names (after lose comas).
        if config == '':
            continue 
        print("Info: Parsing the {} configuration file".format(config))

        # Check if it was already loaded.
        if config in configs_parsed:
            print('Warning: Configuration file {} already parsed - skipping'.format(config))
            continue

        # Check if file exists.
        if not os.path.isfile(config):
            print('Error: Configuration file {} does not exist'.format(config))
            #raise Exception('Error: Configuration file {} does not exist'.format(config))
            exit(-1)

        try:
            # Open file and get parameter dictionary.
            with open(config, 'r') as stream:
                param_dict = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print("Error: Couldn't properly parse the {} configuration file".format(config))
            print('yaml.YAMLERROR:', e)
            exit(-1)

        # Remember that we loaded that config.
        configs_parsed.append(config)

        # Check if there are any default configs to load.
        if 'default_configs' in param_dict:
            # If there are - recursion!
            configs_parsed = recurrent_config_parse(param_dict['default_configs'], configs_parsed)

    # Done, return list of loaded configs. 
    return configs_parsed



def torch_summarize(model, show_weights=True, show_parameters=True, show_total_parameters=True):

    """Summarizes torch model by showing trainable/non-trainable parameters and weights.
    
        :param show_weights: Boolean that control if weights will be shown.
        :param show_parameters: Boolean that control if the layer parameters (trainable + non-trainable) will be shown.
        :param show_total_parameters: Boolean that control if the total number of parameters will be shown.
        
    """
    #add name of the current module
    tmpstr = model.__class__.__name__ + ' (\n'

    #initialize total number non-trainable and trainable parameters
    total_params = 0
    total_trainable_params = 0

    #iterate other all modules
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        #get all needed parameters

        #all parameters
        params = sum([np.prod(p.size()) for p in module.parameters()])
        total_params += params

        #trainable parameters
        mod_parameters = filter(lambda p: p.requires_grad, module.parameters())
        trainable_params = sum([np.prod(p.size()) for p in mod_parameters])
        total_trainable_params += trainable_params

        #weights
        weights = tuple([tuple(p.size()) for p in module.parameters()])


        #build a giant string text to summarize all parameters
        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
            tmpstr += ', trainable_parameters={}'.format( trainable_params)
        tmpstr += '\n'

    #add the total number of parameters at the end (reset at every module)
    if show_total_parameters:
        tmpstr += 'total_parameters={}'.format(total_params)
        tmpstr += '\n'
        tmpstr += 'total_trainable_parameters={}'.format(total_trainable_params)
        tmpstr += '\n'
        tmpstr += 'non_trainable_parameters={}'.format(total_params-total_trainable_params)

    tmpstr += '\n'
    tmpstr = tmpstr + ')'

    return tmpstr