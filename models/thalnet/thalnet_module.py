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

"""
thalnet_module.py: defines a module in the ThalNet architecture"""

__author__ = "Younes Bouhadjar & Vincent Marois"

import torch
from torch import nn
from utils.app_state import AppState

from models.controllers.controller_factory import ControllerFactory


class ThalnetModule(nn.Module):
    """
    Implements a ``ThalNet`` module.

    """

    def __init__(self,
                 center_size,
                 context_size,
                 center_size_per_module,
                 input_size,
                 output_size):
        """
        Constructor of the ``ThalnetModule``.

        :param input_size: size of the input sequences
        :type input_size: int

        :param output_size: size of the produced output sequences
        :type output_size: int

        :param center_size: Size of the center of the model.
        :type center_size: int

        :param center_size_per_module:  Size of the center slot allocated to each module.
        :type center_size_per_module: int


        """
        # call base constructor
        super(ThalnetModule, self).__init__()

        self.center_size = center_size
        self.context_size = context_size
        self.center_size_per_module = center_size_per_module

        self.output_size = output_size
        self.input_size = input_size

        # Reading mechanism
        self.fc_context = nn.utils.weight_norm(
            nn.Linear(self.center_size, self.context_size), name='weight')

        # Parameters needed for the controller
        self.input_context_size = self.input_size + self.context_size
        self.controller_hidden_size = self.output_size + self.center_size_per_module
        self.controller_type = 'FFGRUController'
        self.non_linearity = ''

        # Set module
        controller_params = {
            "name": self.controller_type,
            "input_size": self.input_context_size,
            "output_size": self.controller_hidden_size,
            "num_layers": 1,
            "non_linearity": self.non_linearity,
            "ff_output_size": center_size_per_module
        }

        self.controller = ControllerFactory.build_controller(controller_params)

    def init_state(self, batch_size):
        """
        Initialize the state of a ``ThalNet`` module.

        :param batch_size: batch size
        :type batch_size: int

        :return: center_state_per_module, tuple_controller_states

        """

        dtype = AppState().dtype

        # module state initialisation
        tuple_controller_states = self.controller.init_state(batch_size)

        # center state initialisation
        center_state_per_module = torch.randn(
            (batch_size, self.center_size_per_module)).type(dtype)

        return center_state_per_module, tuple_controller_states

    def forward(self, inputs, prev_center_state, prev_tuple_controller_state):
        """
        Forward pass of a ``ThalnetModule``.

        :param inputs: input sequences.
        :type inputs: torch.tensor

        :param prev_center_state: previous center state
        :type prev_center_state: torch.tensor

        :param prev_tuple_controller_state: previous tuple controller state
        :type prev_tuple_controller_state: tuple

        :return: output, center_feature_output, tuple_ctrl_state

        """
        if inputs is not None:
            if len(inputs.size()) <= 1 or len(inputs.size()) >= 4:
                print('check inputs size of thalnet cell')
                exit(-1)

            if len(inputs.size()) == 3:
                # inputs_size : [batch_size, num_channel, input_size]
                # select channel
                inputs = inputs[:, 0, :]

        # get the context_input and the inputs of the module
        context_input = self.fc_context(prev_center_state)
        inputs = torch.cat((inputs, context_input),
                           dim=1) if self.input_size else context_input

        # Apply the controller
        module_state, tuple_ctrl_state = self.controller(
            inputs, prev_tuple_controller_state)

        output, center_feature_output = torch.split(
            module_state, [self.output_size, self.center_size_per_module],
            dim=1) if self.output_size else(
            None, module_state)

        return output, center_feature_output, tuple_ctrl_state
