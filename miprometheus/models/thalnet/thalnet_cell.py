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
thalnet_cell: A ThalNetCell, constituted of ThalNet modules. It operates on a single word.
"""
__author__ = "Younes Bouhadjar & Vincent Marois"

import torch
from torch.nn import Module
from miprometheus.models.thalnet.thalnet_module import ThalnetModule


class ThalNetCell(Module):
    """
    Implementation of the ``ThalNetCell``, iterating over one sequence element at a time.

    It is constituted of several ``ThalNetModule``.

    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 context_input_size: int,
                 center_size_per_module: int,
                 num_modules: int):
        """
        Constructor of the ``ThalNetCell`` class.

        :param input_size: size of the input sequences
        :type input_size: int

        :param output_size: size of the produced output sequences
        :type output_size: int

        :param context_input_size: context input size
        :type context_input_size: int

        :param center_size_per_module:  Size of the center slot allocated to each module.
        :type center_size_per_module: int

        :param num_modules: number of modules to constitute the cell.
        :type num_modules: int

        """
        # Call base class inits here.
        super(ThalNetCell, self).__init__()

        self.context_input_size = context_input_size
        self.input_size = input_size
        self.output_size = output_size
        self.center_size = num_modules * center_size_per_module
        self.center_size_per_module = center_size_per_module
        self.num_modules = num_modules

        # init module-center cell
        self.modules_thalnet = torch.nn.ModuleList()

        self.modules_thalnet.append(
            ThalnetModule(
                center_size=self.center_size,
                context_size=self.context_input_size,
                center_size_per_module=self.center_size_per_module,
                input_size=self.input_size,
                output_size=0))

        self.modules_thalnet.extend(
            [
                ThalnetModule(
                    center_size=self.center_size,
                    context_size=self.context_input_size,
                    center_size_per_module=self.center_size_per_module,
                    input_size=0,
                    output_size=self.output_size if i == self.num_modules -
                    1 else 0) for i in range(
                    1,
                    self.num_modules)])

    def init_state(self, batch_size):
        """
        Initialize the state of ``ThalNet``.

        :param batch_size: batch size
        :type batch_size: int

        :return: Initialized states of the ThalNet cell.

        """

        # module and center state initialisation
        states = [self.modules_thalnet[i].init_state(
            batch_size) for i in range(self.num_modules)]

        return states

    def forward(self, inputs, prev_state):
        """
        forward run of the ``ThalNetCell``.

        :param inputs: inputs at time t, [batch_size, input_size]
        :type inputs: torch.tensor

        :param prev_state: previous state [batch_size, state_size]
        :type prev_state: torch.tensor

        :return:

            - states [batch_size, state_size]
            - prediction [batch_size, output_size]

        """
        prev_center_states = [prev_state[i][0]
                              for i in range(self.num_modules)]
        prev_controller_states = [prev_state[i][1]
                                  for i in range(self.num_modules)]

        # Concatenate all the centers
        prev_center_states = torch.cat(prev_center_states, dim=1)

        states = []
        # run the different modules, they share all the same center
        for module, prev_controller_state in zip(
                self.modules_thalnet, prev_controller_states):
            output, center_feature, module_state = module(
                inputs, prev_center_states, prev_controller_state)
            states.append((center_feature, module_state))

        return output, states
