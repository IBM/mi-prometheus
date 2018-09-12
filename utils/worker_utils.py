#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""worker_utils.py: Contains helper functions for different workers"""
__author__ = "Ryan McAvoy, Tomasz Kornuta, Vincent Albuoy"

import os
import yaml
import numpy as np

import torch
from torch.nn.modules.module import _addindent

from .app_state import AppState


def forward_step(model, problem, episode, stat_col, data_tuple, aux_tuple):
    """
    Function performs a single forward step.

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
    """
    Enables Cuda if available and sets the default data types.

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

    # TODO Add flags to change these
    AppState().set_dtype('float')
    AppState().set_itype('int')


def recurrent_config_parse(configs, configs_parsed):
    """
    Function parses names of configuration files in a recursive mannner, i.e.
    by looking for 'default_config' sections and trying to load and parse those
    files one by one.

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
            print(
                'Warning: Configuration file {} already parsed - skipping'.format(config))
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
            print(
                "Error: Couldn't properly parse the {} configuration file".format(config))
            print('yaml.YAMLERROR:', e)
            exit(-1)

        # Remember that we loaded that config.
        configs_parsed.append(config)

        # Check if there are any default configs to load.
        if 'default_configs' in param_dict:
            # If there are - recursion!
            configs_parsed = recurrent_config_parse(
                param_dict['default_configs'], configs_parsed)

    # Done, return list of loaded configs.
    return configs_parsed




def model_summarize(model):
    """Summarizes torch model by showing trainable/non-trainable parameters and weights.
        Uses recursive_summarize to interate through nested structure of the mode.
            
    """
    #add name of the current module
    summary_str = '\n' + '='*80 + '\n'
    summary_str += 'Model name (Type) \n'
    summary_str += '+ Submodule name (Type) \n'
    summary_str += '    Matrices: [(name, dims), ...]\n'
    summary_str += '    Trainable Params: #\n'
    summary_str += '    Non-trainable Params: #\n'
    summary_str += '='*80 + '\n'
    summary_str += model.name + ' ' + recursive_summarize(model, 1, len(model.name)+1)
    # Sum the model parameters.
    num_total_params = sum([np.prod(p.size()) for p in model.parameters()])
    mod_trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_params = sum([np.prod(p.size()) for p in mod_trainable_params])
    summary_str += '\nTotal Trainable Params: {}\n'.format(num_trainable_params)
    summary_str += 'Total Non-trainable Params: {}\n'.format(num_total_params-num_trainable_params) 
    summary_str += '='*80 + '\n'
    return summary_str

def recursive_summarize(module_, indent_, name_len_):
    # Iterate through children.
    child_lines = []
    for key, module in module_._modules.items():
        child_str = '| '*(indent_-1) + '+ ' + key + ' '
        child_str += recursive_summarize(module, indent_+1, len(key)+1)
        child_lines.append(child_str)

    # "Leaf information". 
    mod_type_name = module_._get_name()
    mod_str = "("+ mod_type_name + ')'
    #mod_str += '-'*(80 - 2*indent_ - name_len_ - len(mod_type_name))
    mod_str += '\n'
    mod_str += ''.join(child_lines)
    # Get leaf weights and number of params - only for leafs!
    if not child_lines:
        # Collect names and dimensions of all (named) params. 
        mod_weights = [(n,tuple(p.size())) for n,p in module_.named_parameters()]
        mod_str += '| '* (indent_ -1) + '  Matrices: {}\n'.format(mod_weights)
        # Sum the parameters.
        num_total_params = sum([np.prod(p.size()) for p in module_.parameters()])
        mod_trainable_params = filter(lambda p: p.requires_grad, module_.parameters())
        num_trainable_params = sum([np.prod(p.size()) for p in mod_trainable_params])
        mod_str += '| '* (indent_ -1) + '  Trainable Params: {}\n'.format(num_trainable_params)
        mod_str += '| '* (indent_ -1) + '  Non-trainable Params: {}\n'.format(num_total_params-num_trainable_params) 
   
    return mod_str