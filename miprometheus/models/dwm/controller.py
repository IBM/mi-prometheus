#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""controller.py: Calls DWM controller """
__author__ = "Younes Bouhadjar"

import torch
from torch import nn

from miprometheus.models.controllers.controller_factory import ControllerFactory


class Controller(nn.Module):
    """
    Implementation of the DWM controller.
    """

    def __init__(self, in_dim, output_units, state_units,
                 read_size, update_size):
        """
        Constructor for the Controller.

        :param in_dim: input size.
        :param output_units: output size.
        :param state_units: state size.
        :param read_size: size of data_gen read from memory
        :param update_size: total number of parameters for updating attention and memory

        """
        super(Controller, self).__init__()

        self.read_size = read_size
        self.update_size = update_size
        self.state_units = state_units

        self.ctrl_in_dim = in_dim + self.read_size
        self.ctrl_in_state_dim = in_dim + state_units + self.read_size

        # Output layer
        self.output_units = output_units

        # State layer dictionary
        self.controller_type = 'RNNController'
        self.non_linearity = 'sigmoid'

        controller_params = {
            "name": self.controller_type,
            "input_size": self.ctrl_in_dim,
            "output_size": self.state_units,
            "num_layers": 1,
            "non_linearity": self.non_linearity
        }

        # State layer
        self.i2s = ControllerFactory.build_controller(controller_params)

        # Update layer
        self.i2u = nn.Linear(self.ctrl_in_state_dim, self.update_size)

        # Output layer
        self.i2o = nn.Linear(self.ctrl_in_state_dim, self.output_units)

    def init_state(self, batch_size):
        """
        Returns 'zero' (initial) state tuple.

        :param batch_size: size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - object of LSTMStateTuple class.

        """

        return self.i2s.init_state(batch_size)

    def forward(self, input, tuple_state_prev, read_data):
        """
        Forward pass of the DWM controller, calculates the output, the hidden
        state and the interface parameters.

        :param input: current input (from time t) [batch_size, in_dim]
        :param tuple_state_prev: contains previous hidden state (from time t-1) [batch_size, state_units]
        :param read_data: read data from memory (from time t) [batch_size, read_size]

        :returns: output: logits represent the prediction [batch_size, output_units]
        :returns: tuple_state: contains new_hidden_state
        :returns: update_data: interface parameters [batch_size, update_size]

        """
        # Concatenate the 3 inputs to controller
        combined = torch.cat((input, read_data), dim=-1)
        combined_with_state = torch.cat(
            (combined, tuple_state_prev.hidden_state), dim=-1)

        # Get the state and update; no activation is applied
        state, tuple_state = self.i2s(combined, tuple_state_prev)

        # Get output with activation
        output = self.i2o(combined_with_state)

        # update attentional parameters and memory update parameters
        update_data = self.i2u(combined_with_state)

        return output, tuple_state, update_data
