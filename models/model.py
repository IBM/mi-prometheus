#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model.py: contains base abstract model for all models"""
__author__ = "Tomasz Kornuta"

import torch
from torch import nn
from abc import abstractmethod

import numpy as np

import logging
logger = logging.getLogger('Model')

from utils.app_state import AppState


class Model(nn.Module):
    """
    Class representing base class of all models.

    Provides basic plotting functionality.

    """

    def __init__(self, params):
        """
        Initializes application state and sets plot if visualization flag is
        turned on.

        :param params: Parameters read from configuration file.

        """
        # Call base class inits here.
        super(Model, self).__init__()

        # Initialize app state.
        self.app_state = AppState()

        # Store pointer to params.
        self.params = params

        # Window in which the data will be ploted.
        self.plotWindow = None

        # Initialization of best loss - as INF.
        self.best_loss = np.inf

        # Flag indicating whether intermediate checkpoints should be saved or
        # not (DEFAULT: False).
        if "save_intermediate" not in params:
            params.add_default_params({"save_intermediate": False})
        self.save_intermediate = params["save_intermediate"]

        # "Default" model name.
        self.name = 'Model'


    def add_statistics(self, stat_col):
        """
        Add statistics to collector.

        EMPTY - To be redefined in inheriting classes.

        :param stat_col: Statistics collector.

        """
        pass

    def collect_statistics(self, stat_col, data_tuple, logits):
        """
        Base statistics collection.

        EMPTY - To be redefined in inheriting classes.

        :param stat_col: Statistics collector.
        :param data_tuple: Data tuple containing inputs and targets.
        :param logits: Logits being output of the model.

        """
        pass

    @abstractmethod
    def plot(self, data_tuple, predictions, sample_number=0):
        """
        Plots inputs, targets and predictions, along with model-dependent
        variables.

        Abstract - to be defined in derived classes.

        :param data_tuple: Data tuple containing input and target batches.
        :param predictions: Prediction.
        :param sample_number: Number of sample in batch (DEFAULT: 0)

        """

    def save(self, model_dir, stat_col):
        """
        Generic method saving the model parameters to file. It can be
        overloaded if one needs more control.

        :param model_dir: Directory where the model will be saved.
        :param stat_col: Statistics collector that contain current loss and episode number (and other statistics).
        :return: True if this is the best model that is found till now (considering loss).

        """
        # Get two elementary statistics.
        loss = stat_col['loss']
        episode = stat_col['episode']

        # Checkpoint to be saved.
        chkpt = {
            'name': self.name,
            'state_dict': self.state_dict(),
            'stats': stat_col.statistics
        }

        # for key, value in stat_col.statistics.items():
        #    logger.warning("{}: {}".format(key, value))

        # Save the intermediate checkpoint.
        if self.save_intermediate:
            filename = model_dir + 'model_episode_{:05d}.pt'.format(episode)
            torch.save(chkpt, filename)
            logger.info(
                "Model and statistics exported to checkpoint {}".format(
                    filename))

        # Save the best model.
        if (loss < self.best_loss):
            self.best_loss = loss
            filename = model_dir + 'model_best.pt'
            torch.save(chkpt, filename)
            logger.info(
                "Model and statistics exported to checkpoint {}".format(
                    filename))
            return True
        # Else: that was not the best model.
        return False

    def load(self, checkpoint_file):
        """
        Loads model from the checkpoint file.

        :param checkpoint_file: File containing dictionary with model state and statistics.

        """
        # Load checkpoint
        # This is to be able to load CUDA-trained model on CPU
        chkpt = torch.load(
            checkpoint_file, map_location=lambda storage, loc: storage)

        # Load model.
        self.load_state_dict(chkpt['state_dict'])

        # Print statistics.
        logger.info(
            "Imported {} parameters from checkpoint (episode {}, loss {})".format(
                chkpt['name'],
                chkpt['stats']['episode'],
                chkpt['stats']['loss']))


    def summarize(self):
        """Summarizes model by showing trainable/non-trainable parameters and weights.
            Uses recursive_summarize to interate through nested structure of the mode.

            :param: Model object for which the summary will be created.
                
        """
        #add name of the current module
        summary_str = '\n' + '='*80 + '\n'
        summary_str += 'Model name (Type) \n'
        summary_str += '  + Submodule name (Type) \n'
        summary_str += '      Matrices: [(name, dims), ...]\n'
        summary_str += '      Trainable Params: #\n'
        summary_str += '      Non-trainable Params: #\n'
        summary_str += '='*80 + '\n'
        summary_str += self.recursive_summarize(self, 0, self.name)
        # Sum the model parameters.
        num_total_params = sum([np.prod(p.size()) for p in self.parameters()])
        mod_trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        num_trainable_params = sum([np.prod(p.size()) for p in mod_trainable_params])
        summary_str += '\nTotal Trainable Params: {}\n'.format(num_trainable_params)
        summary_str += 'Total Non-trainable Params: {}\n'.format(num_total_params-num_trainable_params) 
        summary_str += '='*80 + '\n'
        return summary_str

    def recursive_summarize(self, module_, indent_, module_name_):
        """
            Function that recursively inspects the (sub)modules and records their statistics (like names, types, parameters, their numbers etc.) 

            :param module_: Module to be inspected.
            :param indent_: Current indentation level.
            :param module_name_: Name of the module that will be displayed before its type. 

        """
        # Recursivelly inspect the children.
        child_lines = []
        for key, module in module_._modules.items():
            child_lines.append(self.recursive_summarize(module, indent_+1, key))

        # "Leaf information". 
        mod_str = ''
        if indent_ > 0:
            mod_str += '  ' + '| '*(indent_-1) + '+ '
        mod_str += module_name_ + " ("+ module_._get_name() + ')'
        #mod_str += '-'*(80 - len(mod_str))
        mod_str += '\n'
        mod_str += ''.join(child_lines)
        # Get leaf weights and number of params - only for leafs!
        if not child_lines:
            # Collect names and dimensions of all (named) params. 
            mod_weights = [(n,tuple(p.size())) for n,p in module_.named_parameters()]
            mod_str += '  ' + '| '* (indent_) + '  Matrices: {}\n'.format(mod_weights)
            # Sum the parameters.
            num_total_params = sum([np.prod(p.size()) for p in module_.parameters()])
            mod_trainable_params = filter(lambda p: p.requires_grad, module_.parameters())
            num_trainable_params = sum([np.prod(p.size()) for p in mod_trainable_params])
            mod_str += '  ' + '| '* (indent_) + '  Trainable Params: {}\n'.format(num_trainable_params)
            mod_str += '  ' + '| '* (indent_) + '  Non-trainable Params: {}\n'.format(num_total_params-num_trainable_params) 
    
        return mod_str