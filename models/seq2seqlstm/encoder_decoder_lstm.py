#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple Encoder Decoder LSTM tested on serial recall task"""
__author__ = "Vincent Albouy"

from enum import Enum
import torch
from torch import nn
from torch.autograd import Variable
from models.sequential_model import SequentialModel

class EncoderDecoderLSTM(SequentialModel):
    """Simple Encoder Decoder LSTM """
    

    def  __init__(self, params):
        '''
        Constructor. Initializes parameters on the basis of dictionary passed as argument.
        Warning: Class assumes, that the whole batch has the same length, i.e. batch of subsequences 
        becoming input to encoder is of the same length (ends at the same item), the same goes to
        subsequences being input to decoder.

        :param params: Dictionary of parameters.
        '''

        # Call base constructor.
        super(EncoderDecoderLSTM, self).__init__(params)

        # Parse parameters.
        # Set input and output sizes.
        self.input_size_decoder = params["data_bits"]
        self.input_size_encoder = params["data_bits"] + params["control_bits"]

        self.output_size = params['data_bits']
        self.hidden_state_dim = params["hidden_state_dim"]

        # Indices of control bits triggering encoding/decoding.
        self.encoding_bit = params['encoding_bit']  # Def: 0
        self.decoding_bit = params['decoding_bit']  # Def: 1

        # Create the Encoder.
        self.encoder = nn.LSTMCell(self.input_size_encoder, self.hidden_state_dim)

        # Create the Decoder/Solver.
        self.decoder = nn.LSTMCell(self.input_size_decoder, self.hidden_state_dim)

        # Output linear layer.
        self.output = nn.Linear(self.hidden_state_dim, self.output_size)

        self.modes = Enum('Modes', ['Encode', 'Decode'])

    def init_state(self, batch_size, dtype):

        # Initialize the hidden state.
        h_init = Variable(torch.zeros(batch_size, self.hidden_state_dim).type(dtype), requires_grad=False)

        # Initialize the memory cell state.
        c_init = Variable(torch.zeros(batch_size, self.hidden_state_dim).type(dtype), requires_grad=False)

        # Pack and return a tuple.
        return (h_init, c_init)

    def forward(self, data_tuple):

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

            # switch between the encoder and decoder modes. It will stay in this mode till it hits the opposite kind of marker
            if x[0, self.decoding_bit] and not x[0, self.encoding_bit]:
                mode = self.modes.Decode
            elif x[0, self.encoding_bit] and not x[0, self.decoding_bit]:
                mode = self.modes.Encode
            elif x[0, self.encoding_bit] and x[0, self.decoding_bit]:
                print('Error: both encoding and decoding bit were true')
                exit(-1)

            logit = self.output(h)
            logits += [logit]

            if mode == self.modes.Decode:
                h, c = self.decoder(logit, (h, c))
            elif mode == self.modes.Encode:
                h, c = self.encoder(x, (h, c))

        # Stack logits along the temporal (sequence) axis.
        logits = torch.stack(logits, 1)
        return logits
