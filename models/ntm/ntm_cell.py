#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ntm_cell.py: pytorch module implementing single (recurrent) cell of Neural Turing Machine"""
__author__ = "Tomasz Kornuta"

import torch 
import collections
from interface import Interface

# Helper collection type.
_NTMCellStateTuple = collections.namedtuple('NTMCellStateTuple', ('ctrl_state', 'interface_state',  'memory_state', 'read_vector'))

class NTMCellStateTuple(_NTMCellStateTuple):
    """Tuple used by NTM Cells for storing current/past state information"""
    __slots__ = ()


class NTMCell(torch.nn.Module):
    """ Class representing a single NTM cell. """

    def __init__(self, params):
        """ Cell constructor.
        Cell creates controller and interface.
        It also initializes memory "block" that will be passed between states.
            
        :param params: Dictionary of parameters.
        """
        # Call constructor of base class.
        super(NTMCell, self).__init__() 
        
        # Parse parameters.
        # Get input and output  dimensions.
        self.input_size = params["num_control_bits"] + params["num_data_bits"]
        self.output_size = params["num_data_bits"]
        # Get controller hidden state size.
        self.ctrl_hidden_state_size = params['ctrl_hidden_state_size']
        # Get memory parameters.
        self.num_memory_bits = params['num_memory_bits']
        
        # Controller - entity that processes input and produces hidden state of the ntm cell.
        if params["ctrl_type"] == 'ff':
            from feedforward_controller import FeedforwardController            
            self.controller = FeedforwardController(params)
        elif params["ctrl_type"] == 'lstm':
            from lstm_controller import LSTMController            
            self.controller = LSTMController(params)
        else:
            raise ValueError
        
        # Interface - entity responsible for accessing the memory.
        self.interface = Interface(params)

        # Layer that produces output on the basis of... hidden state?
        self.hidden2output = torch.nn.Linear(self.ctrl_hidden_state_size, self.output_size)
        
        
    def init_state(self,  batch_size,  num_memory_addresses):
        """
        Returns 'zero' (initial) state:
        * memory  is reset to random values.
        * read & write weights (and read vector) are set to 1e-6.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :param num_memory_addresses: Number of memory addresses.
        """
        # Initialize controller state.
        ctrl_init_state =  self.controller.init_state(batch_size)

        # Initialize interface state. 
        interface_init_state =  self.interface.init_state(batch_size,  num_memory_addresses)

        # Memory [BATCH_SIZE x MEMORY_BITS x MEMORY_SIZE] 
        init_memory_BxMxS = torch.empty(batch_size,  self.num_memory_bits,  num_memory_addresses)
        torch.nn.init.normal_(init_memory_BxMxS, mean=0.5, std=0.2)
        # Read vector [BATCH_SIZE x MEMORY_SIZE]
        read_vector_BxM = torch.ones((batch_size, num_memory_addresses), dtype=torch.float64)*1e-6
        
        # Pack and return a tuple.
        return NTMCellStateTuple(ctrl_init_state, interface_init_state,  init_memory_BxMxS, read_vector_BxM)


    def forward(self, inputs_BxI,  prev_cell_state):
        """
        Forward function of NTM cell.
        
        :param inputs_BxI: a Tensor of input data of size [BATCH_SIZE  x INPUT_SIZE]
        :param  prev_cell_state: a NTMCellStateTuple tuple, containing previous state of the cell.
        :returns: an output Tensor of size  [BATCH_SIZE x OUTPUT_SIZE] and  NTMCellStateTuple tuple containing current cell state.
        """
        # Unpack previous cell  state.
        (prev_ctrl_state_tuple, prev_interface_state_tuple,  prev_memory_BxMxS, prev_read_vector_BxM) = prev_cell_state
        
        # Execute controller forward step.
        ctrl_output_BxH,  ctrl_state_tuple = self.controller(inputs_BxI, prev_ctrl_state_tuple)
        
        # Output layer.
        logits_BxO = self.hidden2output(ctrl_output_BxH)

        # TODO:  REMOVE THOSE LINES.
        interface_state_tuple = prev_interface_state_tuple
        memory_BxMxS = prev_memory_BxMxS
        read_vector_BxM = prev_read_vector_BxM
        
        # Pack current cell state.
        cell_state_tuple = NTMCellStateTuple(ctrl_state_tuple, interface_state_tuple,  memory_BxMxS, read_vector_BxM)
        
        # Return logits and current cell state.
        return logits_BxO, cell_state_tuple
    
