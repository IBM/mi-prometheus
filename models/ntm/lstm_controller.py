#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lstm_controller.py: pytorch module implementing wrapper for LSTM controller of NTM."""
__author__ = "Tomasz Kornuta"


import torch 
import collections

# Helper collection type.
_LSTMStateTuple = collections.namedtuple('NTMStateTuple', ('hidden_state', 'cell_state'))

class LSTMStateTuple(_LSTMStateTuple):
    """Tuple used by NTM Cells for storing current/past state information"""
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

        # Get memory parameters - required to calculate proper size of input. :]
        self.num_memory_bits = params['num_memory_bits']
        # Get interface parameters - required by initialization of read vectors. :]
        self.interface_num_read_heads = params['interface_num_read_heads']
        
        # controller_input_size = input_size + read_vector_size * num_read_heads
        concatenated_inputs_size = self.input_size +  self.num_memory_bits*self.interface_num_read_heads

        # Cell that processes input and produces hidden state of controller.
        self.lstm = torch.nn.LSTMCell(concatenated_inputs_size, self.ctrl_hidden_state_size)

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

    def forward(self, inputs_BxI,  prev_read_vectors_BxM_H, prev_state_tuple):
        """
        Controller forward function. 
        
        :param inputs_BxI: a Tensor of input data of size [BATCH_SIZE  x INPUT_SIZE].
        :param prev_read_vectors_BxM_H: List of length H (number of heads) previous read vectors of size [BATCH_SIZE x MEMORY_BITS]
        :param prev_state_tuple: object of LSTMStateTuple class containing previous controller state.
        :returns: outputs a Tensor of size  [BATCH_SIZE x OUTPUT_SIZE] and state tuple - object of LSTMStateTuple class.
        """
        # Unpack previous cell  state - just to make sure that everything is ok...
        (hidden_state,  cell_state) = prev_state_tuple

        # Concatenate inputs with read vectors [BATCH_SIZE x (INPUT + NUM_HEADS * MEMORY_BITS)]
        read_vectors = torch.cat(prev_read_vectors_BxM_H, dim=0)
        concat_input = torch.cat((inputs_BxI,  read_vectors), dim=1)
        
        # Execute LSTM single step.
        hidden_state,  cell_state = self.lstm(concat_input, (hidden_state,  cell_state))

        # Pack current cell state.
        state_tuple = LSTMStateTuple(hidden_state,  cell_state)
        
        # Return hidden_state (as output) and state tuple.
        return hidden_state,  state_tuple
 
