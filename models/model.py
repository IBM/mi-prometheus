#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model.py: contains base abstract model for all models"""
__author__      = "Tomasz Kornuta"

import numpy as np
import logging
import torch

# Add path to main project directory - so we can test the base plot, saving images, movies etc.
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..')) 
from misc.app_state import AppState
from problems.problem import DataTuple


class Model(torch.nn.Module):
    """ Class representing base class of all models.
    Provides basic plotting functionality.
    """
    def __init__(self, params):
        """ Initializes application state and sets plot if visualization flag is turned on."""
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

    def plot(self, data_tuple, predictions):
        """ 
        Empty function.
        :param data_tuple: Data tuple containing input  and target.
        :param predictions: Prediction.
        """
        pass

if __name__ == '__main__':
    # Set logging level.
    logging.basicConfig(level=logging.DEBUG)
    
    # Set visualization.
    AppState().visualize = True
    
    # Test base model.
    params = []
    test = Model(params)
    
    while True:
        # Generate new sequence.
        x = np.random.binomial(1, 0.5, (1,  15))
        y = np.random.binomial(1, 0.5, (1,  15))
        z = np.random.binomial(1, 0.5, (1,  15))
        print(x.shape)
        # Transform to PyTorch.
        x = torch.from_numpy(x).type(torch.FloatTensor)
        y=  torch.from_numpy(y).type(torch.FloatTensor)
        z=  torch.from_numpy(z).type(torch.FloatTensor)
        dt = DataTuple(x, y)
        # Plot it and check whether window was closed or not. 
        if test.plot(dt, z):
            break