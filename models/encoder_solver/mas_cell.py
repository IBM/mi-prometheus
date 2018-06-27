#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mas_cell.py: pytorch module implementing single (recurrent) cell of Memory-Augmented Solver"""
__author__ = "Tomasz Kornuta"

import torch 
import collections

# Set logging level.
import logging
logger = logging.getLogger('MAS-Cell')
logging.basicConfig(level=logging.DEBUG)

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'controllers'))
from controller_factory import ControllerFactory
from misc.app_state import AppState

from models.encoder_solver.mas_interface import MASInterface

# Helper collection type.
_MASCellStateTuple = collections.namedtuple('MASCellStateTuple', ('ctrl_state', 'interface_state',  'memory_state', 'read_vector'))

class MASCellStateTuple(_MASCellStateTuple):
    """Tuple used by MAS Cells for storing current/past state information"""
    __slots__ = ()


class MASCell(torch.nn.Module):
    """ Class representing a single Memory-Augmented Decoder cell. """

    def __init__(self, params):
        """ Cell constructor.
        Cell creates controller and interface.
        Assumes that memory will be initialized by the encoder.
            
        :param params: Dictionary of parameters.
        """
        # Call constructor of base class.
        super(MASCell, self).__init__() 
        
        # Parse parameters.
        # Set input and output sizes. 
        self.input_size = params["num_control_bits"] + params["num_data_bits"]
        try:
            self.output_size  = params['num_output_bits']
        except KeyError:
            self.output_size = params['num_data_bits']

        # Get controller hidden state size.
        self.controller_hidden_state_size = params['controller']['hidden_state_size']


        # Controller - entity that processes input and produces hidden state of the MAS cell.        
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
        self.interface = MASInterface(params)

        # Layer that produces output on the basis of hidden state and vector read from the memory.
        ext_hidden_size = self.controller_hidden_state_size +  params['memory']['num_content_bits']
        self.hidden2output = torch.nn.Linear(ext_hidden_size, self.output_size)


    def init_state(self,  encoder_state):
        """
        Initializes the solver cell state depending on the last state of the encoder.
        Recursively initialization: controller, interface.        

        :param encoder_state: Last state of MAE cell.
        :returns: Initial state tuple - object of MASCellStateTuple class.
        """
        #batch_size,  final_encoder_memory_BxAxC, final_encoder_attention_BxAx1
        # Unpack encoder state tuple.
        (_, enc_interface_state,  enc_memory_BxAxC) = encoder_state
        (enc_attention_BxAx1, _) = enc_interface_state

        # Get dtype.
        #dtype = AppState().dtype
        # Get number of memory addresses.
        batch_size = enc_memory_BxAxC.size(0)
        num_memory_addresses = enc_memory_BxAxC.size(1)

        # Initialize controller state.
        ctrl_init_state =  self.controller.init_state(batch_size)

        # Initialize interface state. 
        interface_init_state =  self.interface.init_state(batch_size,  num_memory_addresses, enc_attention_BxAx1)
        
        # Initialize read vectors - one for every head.
        # Unpack cell state.
        (init_read_attention_BxAx1, _, _, _) = interface_init_state
        
        # Read a vector from memory using the initial attention.
        read_vector_BxC = self.interface.read_from_memory(init_read_attention_BxAx1, enc_memory_BxAxC)
        
        # Pack and return a tuple.
        return MASCellStateTuple(ctrl_init_state, interface_init_state, enc_memory_BxAxC, read_vector_BxC)


    def forward(self, inputs_BxI,  prev_cell_state):
        """
        Forward function of MAS cell.
        
        :param inputs_BxI: a Tensor of input data of size [BATCH_SIZE  x INPUT_SIZE]
        :param  prev_cell_state: a MASCellStateTuple tuple, containing previous state of the cell.
        :returns: an output Tensor of size  [BATCH_SIZE x OUTPUT_SIZE] and  MASCellStateTuple tuple containing current cell state.
        """
        # Unpack previous cell  state.
        (prev_ctrl_state_tuple, prev_interface_state_tuple,  prev_memory_BxAxC, _) = prev_cell_state

        controller_input = inputs_BxI
        # Execute controller forward step.
        ctrl_output_BxH,  ctrl_state_tuple = self.controller(controller_input,  prev_ctrl_state_tuple)
       
        # Execute interface forward step.
        read_vector_BxC, memory_BxAxC, interface_state_tuple = self.interface(ctrl_output_BxH, prev_memory_BxAxC,  prev_interface_state_tuple)
        logger.warning("ctrl_output_BxH {}:\n {}".format(ctrl_output_BxH.size(),  ctrl_output_BxH))  
        logger.warning("read_vector_BxC {}:\n {}".format(read_vector_BxC.size(),  read_vector_BxC))  
        
        # Output layer - takes controller output concateneted with new read vectors.
        ext_hidden = torch.cat([ctrl_output_BxH,  read_vector_BxC], dim=1)
        logits_BxO = self.hidden2output(ext_hidden)
        logger.warning("logits_BxO {}:\n {}".format(logits_BxO.size(),  logits_BxO))  
        
        # Pack current cell state.
        cell_state_tuple = MASCellStateTuple(ctrl_state_tuple, interface_state_tuple,  memory_BxAxC, read_vector_BxC)
        
        # Return logits and current cell state.
        return logits_BxO, cell_state_tuple
    
