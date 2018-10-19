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

"""control_and_params.py: Calls controller and parameter generators """
__author__ = " Ryan L. McAvoy"

import torch
from torch import nn
from models.dnc.param_gen import Param_Generator

from models.controllers.controller_factory import ControllerFactory


class ControlParams(nn.Module):
    def __init__(self, output_size, read_size, params):
        """
        Initialize an Controller.

        :param output_size: output size.
        :param read_size: size of data_gen read from memory
        :param params: dictionary of input parameters

        """
        super(ControlParams, self).__init__()

        self.read_size = read_size

        # Parse parameters.
        # Set input and hidden  dimensions.
        self.input_size = params["input_item_size"]

        ctrl_in_dim = self.input_size + self.read_size

        self.hidden_state_size = params['hidden_state_size']
        # Get memory parameters.
        self.num_memory_bits = params['memory_content_size']

        self.controller_type = params['controller_type']
        self.shift_size = params['shift_size']
        self.num_reads = params['num_reads']
        self.num_writes = params['num_writes']
        self.non_linearity = params['non_linearity']

        # TODO Make Multilayered LSTM controller in the vein of the DNC paper
        # State layer
        controller_params = {
            "name": self.controller_type,
            "input_size": ctrl_in_dim,
            "output_size": self.hidden_state_size,
            "num_layers": 1,
            "non_linearity": self.non_linearity
        }

        self.state_gen = ControllerFactory.build_controller(controller_params)

        self.output_gen = nn.Linear(self.hidden_state_size, output_size)

        # Update layer
        self.param_gen = Param_Generator(
            self.hidden_state_size,
            word_size=self.num_memory_bits,
            num_reads=self.num_reads,
            num_writes=self.num_writes,
            shift_size=self.shift_size)

    def init_state(self, batch_size):
        """
        Returns 'zero' (initial) state tuple.

        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - object of LSTMStateTuple class.

        """
        return self.state_gen.init_state(batch_size)

    def forward(self, inputs, prev_ctrl_state_tuple, read_data):
        """
        Calculates the output, the hidden state and the controller parameters.

        :param inputs: Current input (from time t)  [BATCH_SIZE x INPUT_SIZE]
        :param read_data: data read from memory (from time t-1)  [BATCH_SIZE x num_data_bits]
        :param prev_ctrl_state_tuple: Tuple of states of controller (from time t-1)
        :return: Tuple [output, hidden_state, update_data] (update_data contains all of the controller parameters)

        """
        # Concatenate the 2 inputs to controller
        combined = torch.cat((inputs, read_data), dim=-1)

        hidden_state, ctrl_state_tuple = self.state_gen(
            combined, prev_ctrl_state_tuple)

        output = self.output_gen(hidden_state)

        update_data = self.param_gen(hidden_state)

        return output, ctrl_state_tuple, update_data
