#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ntm_cell.py: pytorch module implementing single (recurrent) cell of Neural Turing Machine"""
__author__ = "Tomasz Kornuta"

import torch 
import collections
from ntm_interface import NTMInterface
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'controllers'))
from controller_factory import ControllerFactory

# Helper collection type.
_NTMCellStateTuple = collections.namedtuple('NTMCellStateTuple', ('ctrl_state', 'interface_state',  'memory_state', 'read_vectors'))

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
        self.controller_hidden_state_size = params['controller']['hidden_state_size']
        
        # Get memory parameters - required by initialization of read vectors. :]
        self.num_memory_content_bits = params['memory']['num_content_bits']
        # Get interface parameters - required by initialization of read vectors. :]
        self.interface_num_read_heads = params['interface']['num_read_heads']

        # Controller - entity that processes input and produces hidden state of the ntm cell.        
        # controller_input_size = input_size + read_vector_size * num_read_heads
        ext_controller_inputs_size = self.input_size +  self.num_memory_content_bits*self.interface_num_read_heads
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
        self.interface = NTMInterface(params)

        # Layer that produces output on the basis of... hidden state?
        ext_hidden_size = self.controller_hidden_state_size +  self.num_memory_content_bits*self.interface_num_read_heads
        self.hidden2output = torch.nn.Linear(ext_hidden_size, self.output_size)
        
        
    def init_state(self,  batch_size,  num_memory_addresses,  dtype):
        """
        Returns 'zero' (initial) state:
        * memory  is reset to random values.
        * read & write weights are set to 1e-6.
        * read_vectors are initialize as 0s.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :param num_memory_addresses: Number of memory addresses.
        :param dtype: dtype of the matrix denoting the device placement (CPU/GPU).
       :returns: Initial state tuple - object of NTMCellStateTuple class.
        """
        # Initialize controller state.
        ctrl_init_state =  self.controller.init_state(batch_size,  dtype)

        # Initialize interface state. 
        interface_init_state =  self.interface.init_state(batch_size,  num_memory_addresses,  dtype)

        # Memory [BATCH_SIZE x MEMORY_ADDRESSES x CONTENT_BITS] 
        init_memory_BxAxC = torch.zeros(batch_size,  num_memory_addresses,  self.num_memory_content_bits).type(dtype)
        #init_memory_BxAxC = torch.empty(batch_size,  num_memory_addresses,  self.num_memory_content_bits).type(dtype)
        #torch.nn.init.normal_(init_memory_BxAxC, mean=0.5, std=0.2)
        
        # Initialize read vectors - one for every head.
        # Unpack cell state.
        (init_read_state_tuples,  init_write_state_tuple) = interface_init_state
        (init_write_attention_BxAx1, _, _, _) = init_write_state_tuple
        (init_read_attentions_BxAx1_H, _, _, _) = zip(*init_read_state_tuples)
        
        read_vectors_BxC_H = []
        for h in range(self.interface_num_read_heads):
            # Read vector [BATCH_SIZE x CONTENT_BITS]
            #read_vectors_BxC_H.append(torch.zeros((batch_size, self.num_memory_content_bits)).type(dtype))
            # Read vectors from memory using the initial attention.
            read_vectors_BxC_H.append(self.interface.read_from_memory(init_read_attentions_BxAx1_H[h],  init_memory_BxAxC))
        
        # Pack and return a tuple.
        return NTMCellStateTuple(ctrl_init_state, interface_init_state,  init_memory_BxAxC, read_vectors_BxC_H)


    def forward(self, inputs_BxI,  prev_cell_state):
        """
        Forward function of NTM cell.
        
        :param inputs_BxI: a Tensor of input data of size [BATCH_SIZE  x INPUT_SIZE]
        :param  prev_cell_state: a NTMCellStateTuple tuple, containing previous state of the cell.
        :returns: an output Tensor of size  [BATCH_SIZE x OUTPUT_SIZE] and  NTMCellStateTuple tuple containing current cell state.
        """
        # Unpack previous cell  state.
        (prev_ctrl_state_tuple, prev_interface_state_tuple,  prev_memory_BxAxC, prev_read_vectors_BxC_H) = prev_cell_state

        # Concatenate inputs with previous read vectors [BATCH_SIZE x (INPUT + NUM_HEADS * MEMORY_CONTENT_BITS)]
        #print("prev_read_vectors_BxC_H =", prev_read_vectors_BxC_H[0].size())
        prev_read_vectors = torch.cat(prev_read_vectors_BxC_H, dim=1)
        #print("inputs_BxI =", inputs_BxI.size())
        #print("prev_read_vectors =", prev_read_vectors.size())
        controller_input = torch.cat((inputs_BxI,  prev_read_vectors ), dim=1)

        # Execute controller forward step.
        ctrl_output_BxH,  ctrl_state_tuple = self.controller(controller_input,  prev_ctrl_state_tuple)
       
        # Execute interface forward step.
        read_vectors_BxC_H,  memory_BxAxC, interface_state_tuple = self.interface(ctrl_output_BxH, prev_memory_BxAxC,  prev_interface_state_tuple)
        
        # Output layer - takes controller output concateneted with new read vectors.
        read_vectors = torch.cat(read_vectors_BxC_H, dim=1)
        ext_hidden = torch.cat((ctrl_output_BxH,  read_vectors ), dim=1)
        logits_BxO = self.hidden2output(ext_hidden)
        
        # Pack current cell state.
        cell_state_tuple = NTMCellStateTuple(ctrl_state_tuple, interface_state_tuple,  memory_BxAxC, read_vectors_BxC_H)
        
        # Return logits and current cell state.
        return logits_BxO, cell_state_tuple
    
