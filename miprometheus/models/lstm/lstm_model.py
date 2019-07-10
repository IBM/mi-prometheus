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

"""lstm_model.py: File containing Long Short-Term Memory model class."""
__author__ = "Alexis Asseman, Tomasz Kornuta"

import torch

from miprometheus.models.sequential_model import SequentialModel


class LSTM(SequentialModel):
    """
    Class implementing the Long Short-Term Memory model.

    """
    def __init__(self, params, problem_default_values_={}):
        """
        Constructor. Initializes parameters on the basis of dictionary passed
        as argument.

        :param params: Local view to the Parameter Regsitry ''model'' section.

        :param problem_default_values_: Dictionary containing key-values received from problem.

        """
        super(LSTM, self).__init__(params)

        print('Params')
        print(dict(params))
        print('Problem default values')
        print(dict(problem_default_values_))
        input('Pause')


        # Parse default values received from problem.
        self.params.add_default_params({
            'input_item_size': problem_default_values_['input_item_size'],
            'output_item_size': problem_default_values_['output_item_size']
            })

        self.input_item_size = params["input_item_size"]
        self.output_item_size = params["output_item_size"]

        self.hidden_state_size = params["hidden_state_size"]
        self.num_layers = params["num_layers"]
        assert self.num_layers > 0, "Number of LSTM layers should be > 0"

        # Create the stacked LSTM.
        self.lstm_layers = torch.nn.ModuleList()
        # First layer.
        self.lstm_layers.append(torch.nn.LSTMCell(
            self.input_item_size, self.hidden_state_size))
        # Following, stacked layers.
        self.lstm_layers.extend(
            [torch.nn.LSTMCell(self.hidden_state_size, self.hidden_state_size)
             for _ in range(1, self.num_layers)])
        # Output linear layer.
        self.linear = torch.nn.Linear(self.hidden_state_size, self.output_item_size)

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

        # Create the hidden state tensors
        h = [
            torch.zeros(
                batch_size,
                self.hidden_state_size,
                requires_grad=False).type(dtype) 
            for _ in range(self.num_layers)]

        # Create the internal state tensors
        c = [
            torch.zeros(
                batch_size,
                self.hidden_state_size,
                requires_grad=False).type(dtype) for _ in range(
                self.num_layers)]

        outputs = []
        # Process items one-by-one.
        for item in inputs_BxSxI.chunk(inputs_BxSxI.size(1), dim=1):
            h[0], c[0] = self.lstm_layers[0](item.squeeze(1), (h[0], c[0]))
            for i in range(1, self.num_layers):
                h[i], c[i] = self.lstm_layers[i](h[i - 1], (h[i], c[i]))

            out = self.linear(h[-1])
            outputs += [out]

        outputs = torch.stack(outputs, 1)
        return outputs
