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

"""rnn_controller.py: pytorch module implementing wrapper for RNN controller of NTM."""
__author__ = "Ryan L. McAvoy"

import torch
import collections
from torch.nn import Module

from miprometheus.utils.app_state import AppState

_RNNStateTuple = collections.namedtuple('RNNStateTuple', ('hidden_state'))


class RNNStateTuple(_RNNStateTuple):
    """
    Tuple used by LSTM Cells for storing current/past state information.
    """
    __slots__ = ()


class RNNController(Module):
    """
    A wrapper class for a feedforward controller?

    TODO: Doc needs update!

    """
    def __init__(self, params):
        """
        Constructor for a RNN.

        :param params: Dictionary of parameters.

        """

        self.input_size = params["input_size"]
        self.ctrl_hidden_state_size = params["output_size"]
        #self.hidden_state_dim = params["hidden_state_dim"]
        self.non_linearity = params["non_linearity"]
        self.num_layers = params["num_layers"]
        assert self.num_layers > 0, "Number of layers should be > 0"

        super(RNNController, self).__init__()
        full_size = self.input_size + self.ctrl_hidden_state_size
        self.rnn = torch.nn.Linear(full_size, self.ctrl_hidden_state_size)

    def init_state(self, batch_size):
        """
        Returns 'zero' (initial) state tuple.

        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - object of RNNStateTuple class.

        """
        # Initialize LSTM hidden state [BATCH_SIZE x CTRL_HIDDEN_SIZE].
        dtype = AppState().dtype
        hidden_state = torch.zeros(
            (batch_size,
             self.ctrl_hidden_state_size),
            requires_grad=False).type(dtype)

        return RNNStateTuple(hidden_state)

    def forward(self, inputs, prev_hidden_state_tuple):
        """
        Controller forward function.

        :param inputs: a Tensor of input data of size [BATCH_SIZE  x INPUT_SIZE] (generally the read data and input word concatenated)
        :param prev_state_tuple: Tuple of the previous hidden state
        :returns: outputs a Tensor of size  [BATCH_SIZE x OUTPUT_SIZE] and an RNN state tuple.

        """

        h = prev_hidden_state_tuple[0]
        combo = torch.cat((inputs, h), dim=-1)
        hidden_state = self.rnn(combo)

        if self.non_linearity == "sigmoid":
            hidden_state = torch.nn.functional.sigmoid(hidden_state)
        elif self.non_linearity == "tanh":
            hidden_state = torch.nn.functional.tanh(hidden_state)
        elif self.non_linearity == "relu":
            hidden_state = torch.nn.functional.relu(hidden_state)

        return hidden_state, RNNStateTuple(hidden_state)
