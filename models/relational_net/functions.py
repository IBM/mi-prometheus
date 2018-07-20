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

"""functions.py: contains implementations of g_theta & f_phi for the Relational Network."""
__author__      = "Vincent Marois"

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from misc.app_state import AppState
app_state = AppState()


class g_theta(nn.Module):
    """
    Implementation of the g_theta MLP used in the Relational Network model. For recall, the role of g_theta is to
    infer the ways in which 2 regions of the CNN feature maps are related, or if they are even related at all
    """

    def __init__(self, params):
        # call base constructor
        super(g_theta, self).__init__()

        self.input_size = params['input_size']

        self.g_fc1 = nn.Linear(in_features=self.input_size, out_features=256)
        self.g_fc2 = nn.Linear(in_features=256, out_features=256)
        self.g_fc3 = nn.Linear(in_features=256, out_features=256)
        self.g_fc4 = nn.Linear(in_features=256, out_features=256)

    def forward(self, inputs):
        """
        forward pass of the g_theta MLP.
        :param inputs: tensor of shape [batch_size, input_size], should represent the pairs of regions (in the CNN
        feature maps) cat with the question encoding.

        :return: tensor of shape [batch_size, 256]
        """

        x = self.g_fc1(inputs)
        x = F.relu(x)

        x = self.g_fc2(x)
        x = F.relu(x)

        x = self.g_fc3(x)
        x = F.relu(x)

        x = self.g_fc4(x)
        x = F.relu(x)

        return x


class f_phi(nn.Module):
    """
        Implementation of the f_phi MLP used in the Relational Network model. For recall, the role of f_phi is to
        produce the probability distribution over all possible answers.
        """

    def __init__(self, params):
        # call base constructor
        super(f_phi, self).__init__()

        self.output_size = params['output_size']

        self.f_fc1 = nn.Linear(in_features=256, out_features=256)
        self.f_fc2 = nn.Linear(in_features=256, out_features=256)
        self.f_fc3 = nn.Linear(in_features=256, out_features=self.output_size)

    def forward(self, inputs):
        """
        forward pass of the f_phi MLP.
        :param inputs: tensor of shape [batch_size, 256], should represent the element-wise sum of the outputs of
        g_theta.

        :return: tensor of shape [batch_size, 256]
        """

        x = self.f_fc1(inputs)
        x = F.relu(x)

        x = self.f_fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        x = self.f_fc3(x)

        return x


if __name__ == '__main__':
    """Unit Tests for g_theta & f_phi."""
    input_size = (24+2)*2+13
    batch_size = 64
    inputs = np.random.binomial(1, 0.5, (batch_size, 3, input_size))
    inputs = torch.from_numpy(inputs).type(app_state.dtype)

    params_g = {'input_size': input_size}
    g_theta = g_theta(params_g)

    g_outputs = g_theta(inputs)
    print('g_outputs:', g_outputs.shape)

    output_size = 29
    params_f = {'output_size': output_size}
    f_phi = f_phi(params_f)

    f_outputs = f_phi(g_outputs)
    print('f_outputs:', f_outputs.shape)
