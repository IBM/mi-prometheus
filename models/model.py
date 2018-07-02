#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model.py: contains base abstract model for all models"""
__author__      = "Tomasz Kornuta"

import torch
from torch import nn
from abc import ABCMeta, abstractmethod

import logging
logger = logging.getLogger('Model')

# Add path to main project directory - so we can test the base plot, saving images, movies etc.
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..')) 
from misc.app_state import AppState


class Model(nn.Module):
    """ Class representing base class of all models.
    Provides basic plotting functionality.
    """
    def __init__(self, params):
        """ 
        Initializes application state and sets plot if visualization flag is turned on.

        :param params: Parameters read from configuration file.
        """
        # Call base class inits here.
        super(Model, self).__init__()

        # WARNING: at that moment AppState must be initialized and flag must be set. Otherwise the object plot won't be created.
        # SOLUTION: if application is supposed to show dynamic plot, set flag to True before constructing the model! (and set to False right after if required)
        self.app_state = AppState()

        # Window in which the data will be ploted.
        self.plotWindow = None

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
    def plot(self, data_tuple, predictions, sample_number = 0):
        """
        Plots inputs, targets and predictions, along with model-dependent variables. 
        Abstract - to be defined in derived classes. 

        :param data_tuple: Data tuple containing input and target batches.
        :param predictions: Prediction.
        :param sample_number: Number of sample in batch (DEFAULT: 0) 
        """

    def save(self, model_dir, episode):
        """
        Generic method saving the model parameters to file.
        It can be overloaded if one needs more control.

        :param model_dir: Directory where the model will be saved.
        :param episode: Episode number used as model identifier.
        :returns: False if saving was successful (TODO: implement true condition if there was an error)
        """
        model_filename = 'model_episode_{:05d}.pth.tar'.format(episode)
        torch.save(self.state_dict(), model_dir + model_filename)
        logger.info("Model exported to checkpoint {}".format(model_dir + model_filename))