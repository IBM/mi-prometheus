#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mad_cell.py: pytorch module implementing single (recurrent) cell of Memory-Augmented Decoder"""
__author__ = "Tomasz Kornuta"

import torch 
import collections
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'controllers'))
from controller_factory import ControllerFactory
from misc.app_state import AppState

from models.encoder_solver.mad_interface import MADInterface

# Helper collection type.
_MADCellStateTuple = collections.namedtuple('MADCellStateTuple', ('ctrl_state', 'interface_state',  'memory_state', 'read_vector'))

class MADCellStateTuple(_MADCellStateTuple):
    """Tuple used by MAD Cells for storing current/past state information"""
    __slots__ = ()


class MADCell(torch.nn.Module):
    """ Class representing a single Memory-Augmented Decoder cell. """

    def __init__(self, params):
        """ Cell constructor.
        Cell creates controller and interface.
        Assumes that memory will be initialized by the encoder.
            
        :param params: Dictionary of parameters.
        """
        # Call constructor of base class.
        super(MADCell, self).__init__() 
        
        # Parse parameters.
        # Set input and output sizes. 
        self.input_size = params["num_control_bits"] + params["num_data_bits"]
        try:
            self.output_size  = params['num_output_bits']
        except KeyError:
            self.output_size = params['num_data_bits']

        # Get controller hidden state size.
        self.controller_hidden_state_size = params['controller']['hidden_state_size']
        
        # Get interface parameters - required by initialization of read vectors. :]
        self.interface_num_read_heads = params['interface']['num_read_heads']

        # Controller - entity that processes input and produces hidden state of the MAD cell.        
        ext_controller_inputs_size = self.input_size 

        # Create dictionary wirh controller parameters.
        controller_params = {
           "name":  params['controller']['name'],
           "input_size": ext_controller_inputs_size,
           "output_size": self.controller_hidden_state_size,
           "non_linearity": params['controller']['non_linearity'], 
           "num_layers": params['controller']['num_layers']
        }
        # Build the controller.
        self.controller = ControllerFactory.build_model(controller_params)     

        # Interface - entity responsible for accessing the memory.
        self.interface = MADInterface(params)
       
        
    def init_state(self,  batch_size,  init_memory_BxAxC):
        """
        Returns 'zero' (initial) state:
        * memory  is reset to random values.
        * read & write weights are set to 1e-6.
        * read_vectors are initialize as 0s.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :param num_memory_addresses: Number of memory addresses.
        :returns: Initial state tuple - object of MADCellStateTuple class.
        """
        # Get dtype.
        #dtype = AppState().dtype
        # Get number of memory addresses.
        num_memory_addresses = init_memory_BxAxC.size(1)

        # Initialize controller state.
        ctrl_init_state =  self.controller.init_state(batch_size)

        # Initialize interface state. 
        interface_init_state =  self.interface.init_state(batch_size,  num_memory_addresses)
        
        # Initialize read vectors - one for every head.
        # Unpack cell state.
        (init_read_state_tuple) = interface_init_state
        (init_read_attention_BxAxH, _, _, _) = zip(*init_read_state_tuple)
        
        # Read vectors from memory using the initial attention.
        read_vector_BxCxH = self.interface.read_from_memory(init_read_attention_BxAxH, init_memory_BxAxC)
        
        # Pack and return a tuple.
        return MADCellStateTuple(ctrl_init_state, interface_init_state,  init_memory_BxAxC, read_vector_BxCxH)


    def forward(self, inputs_BxI,  prev_cell_state):
        """
        Forward function of MAD cell.
        
        :param inputs_BxI: a Tensor of input data of size [BATCH_SIZE  x INPUT_SIZE]
        :param  prev_cell_state: a MADCellStateTuple tuple, containing previous state of the cell.
        :returns: an output Tensor of size  [BATCH_SIZE x OUTPUT_SIZE] and  MADCellStateTuple tuple containing current cell state.
        """
        # Unpack previous cell  state.
        (prev_ctrl_state_tuple, prev_interface_state_tuple,  prev_memory_BxAxC, prev_read_vectors_BxC_H) = prev_cell_state

        controller_input = inputs_BxI
        # Execute controller forward step.
        ctrl_output_BxH,  ctrl_state_tuple = self.controller(controller_input,  prev_ctrl_state_tuple)
       
        # Execute interface forward step.
        read_vectors_BxC_H,  memory_BxAxC, interface_state_tuple = self.interface(ctrl_output_BxH, prev_memory_BxAxC,  prev_interface_state_tuple)
        
        # Output layer - takes controller output concateneted with new read vectors.
        read_vectors = torch.cat(read_vectors_BxC_H, dim=1)
        ext_hidden = torch.cat((ctrl_output_BxH,  read_vectors ), dim=1)
        logits_BxO = self.hidden2output(ext_hidden)
        
        # Pack current cell state.
        cell_state_tuple = MADCellStateTuple(ctrl_state_tuple, interface_state_tuple,  memory_BxAxC, read_vectors_BxC_H)
        
        # Return logits and current cell state.
        return logits_BxO, cell_state_tuple
    
