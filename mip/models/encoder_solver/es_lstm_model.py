#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""es_lstm_model.py: Neural Network implementing Encoder-Decoder/Solver architecture"""
__author__ = "Tomasz Kornuta"

from enum import Enum
import torch
from torch import nn

from mip.models.sequential_model import SequentialModel


class EncoderSolverLSTM(SequentialModel):
    """
    Class representing the Encoder-Solver architecture using LSTM cells as both
    encoder and solver modules.
    """

    def __init__(self, params, problem_default_values_={}):
        """
        Constructor. Initializes parameters on the basis of dictionary passed
        as argument.

        :param params: Local view to the Parameter Regsitry ''model'' section.

        :param problem_default_values_: Dictionary containing key-values received from problem.

        """
        # Call base constructor. Sets up default values etc.
        super(EncoderSolverLSTM, self).__init__(params, problem_default_values_)
        # Model name.
        self.name = 'EncoderSolverLSTM'

        # Parse default values received from problem.
        self.params.add_default_params({
            'input_item_size': problem_default_values_['input_item_size'],
            'output_item_size': problem_default_values_['output_item_size'],
            'encoding_bit': problem_default_values_['store_bit'],
            'solving_bit': problem_default_values_['recall_bit']
            })

        self.input_item_size = params["input_item_size"]
        self.output_item_size = params["output_item_size"]

        # Indices of control bits triggering encoding/decoding.
        self.encoding_bit = params['encoding_bit']  # Def: 0
        self.solving_bit = params['solving_bit']  # Def: 1

        self.hidden_state_size = params["hidden_state_size"]

        # Create the Encoder.
        self.encoder = nn.LSTMCell(self.input_item_size, self.hidden_state_size)

        # Create the Decoder/Solver.
        self.solver = nn.LSTMCell(self.input_item_size, self.hidden_state_size)

        # Output linear layer.
        self.output = nn.Linear(self.hidden_state_size, self.output_item_size)

        self.modes = Enum('Modes', ['Encode', 'Solve'])

    def init_state(self, batch_size):
        """
        Returns 'zero' (initial) state.

        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple (hidden, memory cell).

        """
        dtype = self.app_state.dtype

        # Initialize the hidden state.
        h_init = torch.zeros(batch_size, self.hidden_state_size,
                             requires_grad=False).type(dtype)

        # Initialize the memory cell state.
        c_init = torch.zeros(batch_size, self.hidden_state_size,
                             requires_grad=False).type(dtype)

        # Pack and return a tuple.
        return (h_init, c_init)

    def forward(self, data_dict):
        """
        Forward function requires that the data_dict will contain at least "sequences"

        :param data_dict: DataDict containing at least:
            - "sequences": a tensor of input data of size [BATCH_SIZE x LENGTH_SIZE x INPUT_SIZE]

        :returns: Predictions (logits) being a tensor of size  [BATCH_SIZE x LENGTH_SIZE x OUTPUT_SIZE].

        """
        # Get dtype.
        dtype = self.app_state.dtype

        # Unpack dict.
        inputs_BxSxI = data_dict['sequences']

        # Get batch size.
        batch_size = inputs_BxSxI.size(0)

        # Initialize state variables.
        (h, c) = self.init_state(batch_size)

        # Logits container.
        logits = []

        for x in inputs_BxSxI.chunk(inputs_BxSxI.size(1), dim=1):
            # Squeeze x.
            x = x.squeeze(1)

            # Switch between the encoder and decoder modes. It will stay in
            # this mode till it hits the opposite kind of marker
            if x[0, self.solving_bit] and not x[0, self.encoding_bit]:
                mode = self.modes.Solve
            elif x[0, self.encoding_bit] and not x[0, self.solving_bit]:
                mode = self.modes.Encode
            elif x[0, self.encoding_bit] and x[0, self.solving_bit]:
                print('Error: both encoding and decoding bits were true')
                exit(-1)

            if mode == self.modes.Solve:
                h, c = self.solver(x, (h, c))
            elif mode == self.modes.Encode:
                h, c = self.encoder(x, (h, c))

            # Collect logits.
            logit = self.output(h)
            logits += [logit]

        # Stack logits along the temporal (sequence) axis.
        logits = torch.stack(logits, 1)
        return logits
