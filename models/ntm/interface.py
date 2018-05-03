#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lstm_controller.py: pytorch module implementing NTM interface to external memory."""
__author__ = "Tomasz Kornuta"


import torch 
import collections

# Helper collection type.
_InterfaceStateTuple = collections.namedtuple('InterfaceStateTuple', ('read_weights', 'write_weights'))

class InterfaceStateTuple(_InterfaceStateTuple):
    """Tuple used by interface for storing current/past state information"""
    __slots__ = ()


class Interface(torch.nn.Module):
    """Class realizing interface between controller and memory.
    """
    def __init__(self, params):
        """ Constructor.
        
        :param params: Dictionary of parameters.
        """
        # Call constructor of base class.
        super(Interface, self).__init__() 

        # Parse parameters.
        # Set input and hidden  dimensions.
        self.input_size = params["num_control_bits"] + params["num_data_bits"]
        self.ctrl_hidden_state_size = params['ctrl_hidden_state_size']
        # Get memory parameters.
        self.num_memory_bits = params['num_memory_bits']
        # TODO - move memory size somewhere?
        self.num_memory_addresses = params['num_memory_addresses']
        
        # TODO: interface!
        

    def init_state(self,  batch_size):
        """
        Returns 'zero' (initial) state tuple.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - object of InterfaceStateTuple class.
        """
        # Read attention weights [BATCH_SIZE x MEMORY_SIZE]
        read_attention = torch.ones((batch_size, self.num_memory_addresses), dtype=torch.float64)*1e-6
        # Write attention weights [BATCH_SIZE x MEMORY_SIZE]
        write_attention = torch.ones((batch_size, self.num_memory_addresses), dtype=torch.float64)*1e-6

        return InterfaceStateTuple(read_attention,  write_attention)

    def forward(self, ctrl_hidden_state_BxH,  prev_state_tuple):
        """
        Controller forward function. 
        
        :param ctrl_hidden_state_BxH: a Tensor with controller hidden state of size [BATCH_SIZE  x HIDDEN_SIZE]
        :param prev_state_tuple:
        :returns: ... and state tuple - object of LSTMStateTuple class.
        """
        # Unpack previous cell  state - just to make sure that everything is ok...
        (prev_read_attention,  prev_write_attention) = prev_state_tuple
        
        # Execute single step.
        # TODO!

        # TODO:  REMOVE THOSE LINES.
        read_attention = prev_read_attention
        write_attention = prev_write_attention

        # Pack current cell state.
        state_tuple = InterfaceStateTuple(read_attention,  write_attention)
        
        # Return ??? and state tuple.
        return 0,  state_tuple
 
