#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lstm_controller.py: pytorch module implementing wrapper for LSTM controller of NTM."""
__author__ = "Tomasz Kornuta"


import torch 
import collections

# Helper collection type.
_LSTMStateTuple = collections.namedtuple('NTMStateTuple', ('hidden_state', 'cell_state'))

class LSTMStateTuple(_LSTMStateTuple):
    """Tuple used by NTM Cells for storing current state information"""
    __slots__ = ()


class LSTMController(torch.nn.Module):
    """A wrapper class for a recurrent LSTM controller.
    """
    def __init__(self, params):
        """ Constructor.
        
        :param params: Dictionary of parameters.
        """
        # Call constructor of base class.
        super(LSTMController, self).__init__() 

        # Parse parameters.
        # Set input and hidden  dimensions.
        self.input_size = params["num_control_bits"] + params["num_data_bits"]
        self.ctrl_hidden_state_size = params['ctrl_hidden_state_size']

        # Cell that processes input and produces hidden state of controller.
        self.lstm = torch.nn.LSTMCell(self.input_size, self.ctrl_hidden_state_size)

    def init_state(self,  batch_size):
        """
        Returns 'zero' (initial) state tuple.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - object of LSTMStateTuple class.
        """
        # Initialize LSTM hidden state [BATCH_SIZE x CTRL_HIDDEN_SIZE].
        hidden_state = torch.zeros(batch_size, self.ctrl_hidden_state_size, requires_grad=False)
        # Initialize LSTM memory cell [BATCH_SIZE x CTRL_HIDDEN_SIZE].
        cell_state = torch.zeros(batch_size, self.ctrl_hidden_state_size, requires_grad=False)
        
        return LSTMStateTuple(hidden_state,  cell_state)

    def forward(self, inputs_BxI,  prev_state):
        """
        Controller forward function. 
        
        :param inputs_BxI: a Tensor of input data of size [BATCH_SIZE  x INPUT_SIZE]
        :param prev_state: unused - empty tuple () 
        :returns: outputs a Tensor of size  [BATCH_SIZE x OUTPUT_SIZE] and state tuple - object of LSTMStateTuple class.
        """
        # Unpack previous cell  state - just to make sure that everything is ok...
        (hidden_state,  cell_state) = prev_state
        
        # Execute LSTM single step.
        hidden_state,  cell_state = self.lstm(inputs_BxI, (hidden_state,  cell_state))

        # Pack current cell state.
        state = LSTMStateTuple(hidden_state,  cell_state)
        
        # Return hidden_state (as output) and state tuple.
        return hidden_state,  state
 
