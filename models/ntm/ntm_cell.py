#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ntm_cell.py: pytorch module implementing single (recurrent) cell of Neural Turing Machine"""
__author__ = "Tomasz Kornuta"

import torch 
import collections

# Helper collection type.
_NTMStateTuple = collections.namedtuple('NTMStateTuple', ('memory', 'ctrl_state', 'read_vector',  'read_weights', 'write_weights'))

class NTMStateTuple(_NTMStateTuple):
    """Tuple used by NTM Cells for storing current state information"""
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
        # Set input, hidden and output  dimensions.
        self.input_size = params["num_control_bits"] + params["num_data_bits"]
        self.output_size = params["num_data_bits"]
        self.ctrl_hidden_state_size = params['ctrl_hidden_state_size']
        
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
        # TODO

        # Layer that produces output on the basis of... hidden state?
        self.hidden2output = torch.nn.Linear(self.ctrl_hidden_state_size, self.output_size)
        
        
    def init_state(self,  batch_size):
        """
        Returns 'zero' (initial) state:
        * memory  is reset to random values.
        * read & write weights (and read vector) are set to 1e-6.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        """
        # Initialize controller state.
        ctrl_init_state =  self.controller.init_state(batch_size)

        
        # Memory [SLOT_SIZE x NUMBER_OF_SLOTS]
        #memory_SxN = torch.truncated_normal([self.slot_size,  self.number_of_slots], mean=0.5, stddev=0.2)
        # Read vector [BATCH_SIZE x SLOT_SIZE]
        #read_vector = tensor.new_full((self.batch_size, self.slot_size), 1e-6)
        # Read weights [BATCH_SIZE x NUMBER_OF_SLOTS]
        #read_weights = torch.fill([1, self.number_of_slots], 1e-6)
        # Write weights [BATCH_SIZE x NUMBER_OF_SLOTS]
        #write_weights = torch.fill([1, self.number_of_slots], 1e-6)
        # Pack and return a tuple.
        #return NTMStateTuple(memory_SxN, self.controller.zero_state(), read_vector,  read_weights, write_weights)
        return ctrl_init_state


    def forward(self, inputs_BxI,  prev_cell_state):
        """
        Forward function accepts a Tensor of input data of size [BATCH_SIZE  x INPUT_SIZE] and 
        outputs a Tensor of size  [BATCH_SIZE x OUTPUT_SIZE] . 
        """
        # Unpack previous cell  state.
        #(hidden_state,  cell_state) = prev_state
        prev_ctrl_state_tuple = prev_cell_state
        
        # Execute controller forward step.
        ctrl_output_BxH,  ctrl_state_tuple = self.controller(inputs_BxI, prev_ctrl_state_tuple)
        
        # Output layer.
        logits_BxO = self.hidden2output(ctrl_output_BxH)
        
        # Pack current cell state.
        cell_state_tuple = ctrl_state_tuple
        
        # Return logits and current cell state.
        return logits_BxO, cell_state_tuple
    
