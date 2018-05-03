#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ntm_module.py: pytorch module implementing Neural Turing Machine"""
__author__ = "Tomasz Kornuta"

import torch 
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
    
        # Initialize recurrent NTM cell.
        self.cell = NTMCell(params)
        
    def forward(self, inputs_BxSxI):
        """
        Forward function accepts a Tensor of input data of size [BATCH_SIZE x LENGTH_SIZE x INPUT_SIZE] and 
        outputs a Tensor of size  [BATCH_SIZE x LENGTH_SIZE x OUTPUT_SIZE] . 
        """
        # Initialize 'zero' state.
        state = self.cell.init_state(inputs_BxSxI.size(0))

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
    # "Loaded parameters".
    params = {'num_control_bits': 2, 'num_data_bits': 8, 'batch_size': 2, # input and output size
        'ctrl_hidden_state_size': 5, 
        'num_memory_adresses' :10, 'num_memory_bits': 8}
        
    input_size = params["num_control_bits"] + params["num_data_bits"]
    output_size = params["num_data_bits"]
        
    # Create random Tensors to hold inputs and outputs
    seq_length = 5
    x = torch.randn(params['batch_size'], seq_length,   input_size)
    y = torch.randn(params['batch_size'], seq_length,  output_size)

    # Construct our model by instantiating the class defined above
    model = NTM(params)

    # Test forward pass.
    y_pred = model(x)

    print("\n input {}: {}".format(x.size(), x))
    print("\n prediction {}: {}".format(y_pred.size(), y_pred))
    


