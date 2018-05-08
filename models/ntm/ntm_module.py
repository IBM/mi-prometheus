#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ntm_module.py: pytorch module implementing Neural Turing Machine"""
__author__ = "Tomasz Kornuta"

import torch 
import logging
from ntm_cell import NTMCell

class NTM(torch.nn.Module):
    '''  Class representing the Neural Turing Machine module. '''
    def __init__(self, params):
        '''
        Constructor. Initializes parameters on the basis of dictionary of parameters passed as argument.
        
        :param params: Dictionary of parameters.
        '''
        # Call constructor of base class.
        super(NTM, self).__init__() 

        # Parse parameters.
        # It is stored here, but will we used ONLY ONCE - for initialization of memory called from the forward() function.
        self.num_memory_addresses = params['memory']['num_addresses']
        
        # Initialize recurrent NTM cell.
        self.cell = NTMCell(params)
        
    def forward(self, inputs_BxSxI):
        """
        Forward function accepts a Tensor of input data of size [BATCH_SIZE x LENGTH_SIZE x INPUT_SIZE] and 
        outputs a Tensor of size  [BATCH_SIZE x LENGTH_SIZE x OUTPUT_SIZE] . 
        """
        
        # "Data-driven memory size" - temporal solution.
        # Check memory size.
        num_memory_addresses = self.num_memory_addresses
        if num_memory_addresses == -1:
            # Set equal to input sequence length.
            num_memory_addresses = inputs_BxSxI.size(1)
        
        # Initialize 'zero' state.
        state = self.cell.init_state(inputs_BxSxI.size(0),  num_memory_addresses)

        # List of output logits [BATCH_SIZE x OUTPUT_SIZE] of length SEQ_LENGTH
        output_logits_BxO_S = []

        # Divide sequence into chunks of size [BATCH_SIZE x INPUT_SIZE] and process them one by one.
        for input_t in inputs_BxSxI.chunk(inputs_BxSxI.size(1), dim=1):
            # Process one chunk.
            output_BxO,  state = self.cell(input_t.squeeze(1), state)
            # Append to list of logits.
            output_logits_BxO_S += [output_BxO]

        # Stack logits along time axis (1).
        output_logits_BxSxO = torch.stack(output_logits_BxO_S, 1)

        return output_logits_BxSxO


if __name__ == "__main__":
    # Set logging level.
    logging.basicConfig(level=logging.DEBUG)
    # "Loaded parameters".
    params = {'num_control_bits': 2, 'num_data_bits': 8, # input and output size
        'controller': {'name': 'rnn', 'hidden_state_size': 5,  'num_layers': 1, 'non_linearity': 'none'},  # controller parameters
        'interface': {'num_read_heads': 2,  'shift_size': 3},  # interface parameters
        'memory': {'num_addresses' :4, 'num_content_bits': 7} # memory parameters
        }
        
    logger = logging.getLogger('NTM-Module')
    logger.debug("params: {}".format(params))    
        
    input_size = params["num_control_bits"] + params["num_data_bits"]
    output_size = params["num_data_bits"]
        
    seq_length = 1
    batch_size = 2
    
    # Construct our model by instantiating the class defined above
    model = NTM(params)
    
    # Check for different seq_lengts and batch_sizes.
    for i in range(2):
        # Create random Tensors to hold inputs and outputs
        x = torch.randn(batch_size, seq_length,   input_size)
        y = torch.randn(batch_size, seq_length,  output_size)

        # Test forward pass.
        logger.info("------- forward -------")
        y_pred = model(x)

        logger.info("------- result -------")
        logger.info("input {}:\n {}".format(x.size(), x))
        logger.info("target.size():\n {}".format(y.size()))
        logger.info("prediction {}:\n {}".format(y_pred.size(), y_pred))
    
        # Change batch size and seq_length.
        seq_length = seq_length+1
        batch_size = batch_size+1
    


