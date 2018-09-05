#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lstm_controller.py: pytorch module implementing wrapper for feedforward controller."""
__author__ = "Tomasz Kornuta/Ryan L. McAvoy"

import torch

class FeedforwardController(torch.nn.Module):
    """A wrapper class for a feedforward controller.
    """
    def __init__(self, params):
        """ Constructor.
        
        :param params: Dictionary of parameters.
        """
        # Call constructor of base class.
        super(FeedforwardController, self).__init__() 

        # Parse parameters.
        # Set input and hidden  dimensions.
        self.input_size = params["input_size"]
        self.ctrl_hidden_state_size = params["output_size"]

        # Processes input and produces hidden state of the controller.
        self.ff = torch.nn.Linear(self.input_size, self.ctrl_hidden_state_size)

    def init_state(self,  batch_size):
        """
        Returns 'zero' (initial) state tuple - in this case empy tuple.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - empty ().
        """
        return ()

    def forward(self, inputs_BxI,  prev_state_tuple):
        """
        Controller forward function. 
        
        :param inputs_BxI: a Tensor of input data of size [BATCH_SIZE  x INPUT_SIZE]
        :param prev_state_tuple: unused - empty tuple () 
        :returns: outputs a Tensor of size  [BATCH_SIZE x OUTPUT_SIZE] and empty tuple.
        """
        # Execute feedforward pass.
        hidden_state = self.ff(inputs_BxI)
        
        # Return hidden_state (as output) and empty state tuple.
        return hidden_state,  ()
