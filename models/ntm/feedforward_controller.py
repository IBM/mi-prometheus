#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lstm_controller.py: pytorch module implementing wrapper for feedforward controller of NTM."""
__author__ = "Tomasz Kornuta"

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
        self.input_size = params["num_control_bits"] + params["num_data_bits"]
        self.ctrl_hidden_state_size = params['ctrl_hidden_state_size']
        
        # Get memory parameters - required to calculate proper size of input. :]
        self.num_memory_bits = params['num_memory_bits']
        # Get interface parameters - required by initialization of read vectors. :]
        self.interface_num_read_heads = params['interface_num_read_heads']
        
       # controller_input_size = input_size + read_vector_size * num_read_heads
        concatenated_inputs_size = self.input_size +  self.num_memory_bits*self.interface_num_read_heads
 
        # Processes input and produces hidden state of the controller.
        self.ff = torch.nn.Linear(concatenated_inputs_size, self.ctrl_hidden_state_size)
        
    def init_state(self,  batch_size):
        """
        Returns 'zero' (initial) state tuple - in this case empy tuple.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - empty ().
        """
        return ()

    def forward(self, inputs_BxI,  prev_read_vectors_BxM_H, prev_state_tuple):
        """
        Controller forward function. 
        
        :param inputs_BxI: a Tensor of input data of size [BATCH_SIZE  x INPUT_SIZE]
        :param prev_read_vectors_BxM_H: List of length H (number of heads) previous read vectors of size [BATCH_SIZE x MEMORY_BITS]
        :param prev_state_tuple: unused - empty tuple () 
        :returns: outputs a Tensor of size  [BATCH_SIZE x OUTPUT_SIZE] and empty tuple.
        """
        # Concatenate inputs with read vectors.
        read_vectors = torch.cat(prev_read_vectors_BxM_H, dim=0)
        concat_input = torch.cat((inputs_BxI,  read_vectors), dim=1)

        # Execute feedforward pass.
        hidden_state = self.ff(concat_input)

        # Return hidden_state (as output) and empty state tuple.
        return hidden_state,  ()
 
