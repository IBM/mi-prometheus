#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model.py: contains base abstract model for all models"""
__author__      = "Tomasz Kornuta"

import torch
from torch import nn
from abc import ABCMeta, abstractmethod

import numpy as np

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

        # Initialize app state.
        self.app_state = AppState()

        # Store pointer to params.
        self.params = params

        # Window in which the data will be ploted.
        self.plotWindow = None

        # Initialization of best loss - as INF.
        self.best_loss = np.inf

        # Flag indicating whether intermediate checkpoints should be saved or not (DEFAULT: False).
        if "save_intermediate" not in params:
            params.add_default_params({"save_intermediate": False})
        self.save_intermediate = params["save_intermediate"]

        # Model name.
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
    def plot(self, data_tuple, predictions, sample_number = 0):
        """
        Plots inputs, targets and predictions, along with model-dependent variables. 
        Abstract - to be defined in derived classes. 

        :param data_tuple: Data tuple containing input and target batches.
        :param predictions: Prediction.
        :param sample_number: Number of sample in batch (DEFAULT: 0) 
        """

    def save(self, model_dir, stat_col):
        """
        Generic method saving the model parameters to file.
        It can be overloaded if one needs more control.

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

        #for key, value in stat_col.statistics.items():
        #    logger.warning("{}: {}".format(key, value))

        # Save the intermediate checkpoint.                       
        if self.save_intermediate:
            filename = model_dir + 'model_episode_{:05d}.pt'.format(episode)
            torch.save(chkpt, filename)
            logger.info("Model and statistics exported to checkpoint {}".format(filename))

        # Save the best model.
        if (loss < self.best_loss):
            self.best_loss = loss
            filename = model_dir + 'model_best.pt'
            torch.save(chkpt, filename)
            logger.info("Model and statistics exported to checkpoint {}".format(filename))
            return True
        # Else: that was not the best model.
        return False


    def load(self, checkpoint_file):
        """ Loads model from the checkpoint file 
        
        :param checkpoint_file: File containing dictionary with model state and statistics.
        """
        # Load checkpoint
        chkpt = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)  # This is to be able to load CUDA-trained model on CPU

        # Load model.
        self.load_state_dict(chkpt['state_dict'])

        # Print statistics.
        logger.info("Imported {} parameters from checkpoint (episode {}, loss {})".format(chkpt['name'], chkpt['stats']['episode'], chkpt['stats']['loss']))

