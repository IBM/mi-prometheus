#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""maes_module.py: File containing Memory Augmented Encoder-Solver main module."""
__author__ = "Tomasz Kornuta"

from enum import Enum
import torch
from torch import nn
from torch.autograd import Variable
from misc.app_state import AppState
from models.sequential_model import SequentialModel


class MAES(SequentialModel):
    '''
    Class implementing the Memory Augmented Encoder-Solver model. 
    '''

    def __init__(self, params):
        '''
        Constructor. Initializes parameters on the basis of dictionary passed as argument.

        Warning: Class assumes, that the whole batch has the same length, i.e. batch of subsequences 
        becoming input to encoder is of the same length (ends at the same item), the same goes to
        subsequences being input to decoder.
        
        :param params: Dictionary of parameters.
        '''
        # Call base constructor.
        super(MAES, self).__init__(params)

        # Parse parameters.
        # Set input and output sizes. 
        self.input_size = params["control_bits"] + params["data_bits"]
        try:
            self.output_size  = params['output_bits']
        except KeyError:
            self.output_size = params['data_bits']
        self.hidden_state_dim = params["hidden_state_dim"]

        # Indices of control bits triggering encoding/decoding. 
        self.encoding_bit =  params['encoding_bit'] # Def: 0
        self.solving_bit =  params['solving_bit'] # Def: 1

        # Create the Encoder.
        self.encoder = nn.LSTMCell(self.input_size, self.hidden_state_dim) 

        # Create the Decoder/Solver.
        self.solver = nn.LSTMCell(self.input_size, self.hidden_state_dim)

        # Output linear layer.
        self.output = nn.Linear(self.hidden_state_dim, self.output_size)

        self.modes = Enum('Modes', ['Encode', 'Solve'])

    def init_state(self,  batch_size,  dtype):
        """
        Returns 'zero' (initial) state.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :param dtype: dtype of the matrix denoting the device placement (CPU/GPU).
       :returns: Initial state tuple (hidden, memory cell).
        """

        # Initialize the hidden state.
        h_init = Variable(torch.zeros(batch_size, self.hidden_state_dim).type(dtype), requires_grad=False)

        # Initialize the memory cell state.
        c_init = Variable(torch.zeros(batch_size, self.hidden_state_dim).type(dtype), requires_grad=False)
        
        # Pack and return a tuple.
        return (h_init, c_init)

    def forward(self, data_tuple):
        """
        Forward function accepts a tuple consisting of:
         - a tensor of input data of size [BATCH_SIZE x LENGTH_SIZE x INPUT_SIZE] and 
         - a tensor of targets

        :param data_tuple: Tuple containing inputs and targets.
		:returns: Predictions (logits) being a tensor of size  [BATCH_SIZE x LENGTH_SIZE x OUTPUT_SIZE]. 
        """

        # Unpack tuple.
        (inputs, targets) = data_tuple
        batch_size = inputs.size(0)

        # Check if the class has been converted to cuda (through .cuda() method)
        dtype = torch.cuda.FloatTensor if next(self.encoder.parameters()).is_cuda else torch.FloatTensor

        # Initialize state variables.
        (h, c) = self.init_state(batch_size, dtype)

        # Logits container.
        logits = []

        for x in inputs.chunk(inputs.size(1), dim=1):
            # Squeeze x.
            x = x.squeeze(1)

            #switch between the encoder and decoder modes. It will stay in this mode till it hits the opposite kind of marker
            if x[0, self.solving_bit] and not x[0, self.encoding_bit]:
                mode = self.modes.Solve
            elif x[0, self.encoding_bit] and not x[0, self.solving_bit]:
                mode = self.modes.Encode
            elif x[0, self.encoding_bit] and x[0, self.solving_bit]:
                print('Error: both encoding and decoding bit were true')
                exit(-1)

            if mode == self.modes.Solve:
                h, c = self.solver(x, (h, c))
            elif mode == self.modes.Encode:
                h, c = self.encoder(x, (h, c))
                            
            # Collect logits - whatever happens :] (BUT THIS CAN BE EASILY SOLVED - COLLECT LOGITS ONLY IN DECODER!!)
            logit = self.output(h)
            logits += [logit]

        # Stack logits along the temporal (sequence) axis.
        logits = torch.stack(logits, 1)
        return logits

