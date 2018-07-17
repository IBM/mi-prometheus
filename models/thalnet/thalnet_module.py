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

"""thalnet_module.py: defines a class of a module in the ThalNet architecture"""

__author__= "Younes Bouhadjar"

import torch
from torch import nn
from misc.app_state import AppState
# Add path to main project directory - so we can test the base plot, saving images, movies etc.
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..', '..')) 
from models.controllers.controller_factory import ControllerFactory


class ThalnetModule(nn.Module):
    """ @Younes: MODEL DESCRIPTION GOES HERE! """
    def __init__(self,
                 center_size,
                 context_size,
                 center_size_per_module,
                 input_size,
                 output_size):
        super(ThalnetModule, self).__init__()

        self.center_size = center_size
        self.context_size = context_size
        self.center_size_per_module = center_size_per_module

        self.output_size = output_size
        self.input_size = input_size

        # Reading mechanism
        self.fc_context = nn.utils.weight_norm(nn.Linear(self.center_size, self.context_size), name='weight')

        # Parameters needed for the controller
        self.input_context_size = self.input_size + self.context_size
        self.controller_hidden_size = self.output_size + self.center_size_per_module
        self.controller_type = 'ffgru'
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

        self.controller = ControllerFactory.build_model(controller_params)

    def init_state(self, batch_size):

        dtype = AppState().dtype

        # module state initialisation
        tuple_controller_states = self.controller.init_state(batch_size)

        # center state initialisation
        center_state_per_module = torch.randn((batch_size, self.center_size_per_module)).type(dtype)

        return center_state_per_module, tuple_controller_states

    def forward(self, inputs, prev_center_state, prev_tuple_controller_state):
        """
        :return: output, new_center_features, new_module_state
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
        inputs = torch.cat((inputs, context_input), dim=1) if self.input_size else context_input

        # Apply the controller
        module_state, tuple_ctrl_state = self.controller(inputs, prev_tuple_controller_state)

        output, center_feature_output = torch.split(module_state,
                                        [self.output_size, self.center_size_per_module], dim=1) if self.output_size else (None, module_state)

        return output, center_feature_output, tuple_ctrl_state

