#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2018 Kim Seonghyeon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ------------------------------------------------------------------------------
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
write_unit.py: Implementation of the ``WriteUnit`` for the MAC network. Cf https://arxiv.org/abs/1803.03067 \
for the reference paper.
"""
__author__ = "Vincent Marois"

import torch
from torch.nn import Module

from miprometheus.models.mac_sequential.utils_mac import linear


class WriteUnit(Module):
    """
    Implementation of the ``WriteUnit`` of the MAC network.
    """

    def __init__(self, dim, self_attention=False, memory_gate=False):
        """
        Constructor for the ``WriteUnit``.

        :param dim: global 'd' hidden dimension
        :type dim: int

        :param self_attention: whether or not to use self-attention on the previous control states
        :type self_attention: bool

        :param memory_gate: whether or not to use memory gating.
        :type memory_gate: bool

        """

        # call base constructor
        super(WriteUnit, self).__init__()

        # linear layer for the concatenation of ri & mi-1
        self.concat_layer = linear(2 * dim, dim, bias=True)

        # self-attention & memory gating optional initializations
        self.self_attention = self_attention
        self.memory_gate = memory_gate

        if self.self_attention:
            self.attn = linear(dim, 1, bias=True)
            self.mi_sa_proj = linear(dim, dim, bias=True)
            self.mi_info_proj = linear(dim, dim, bias=True)

        if self.memory_gate:
            self.control = linear(dim, 1, bias=True)

    def forward(self, memory_state, read_vector, ctrl_state):
        """
        Forward pass of the ``WriteUnit``.

        :param memory_states: previous memory states, each of shape [batch_size x dim].
        :type memory_states: list

        :param read_vector: current read vector (output of the read unit), shape [batch_size x dim].
        :type read_vector: torch.tensor

        :param ctrl_state: previous control state, each of shape [batch_size x dim].
        :type ctrl_state: list

        :return: current memory state, shape [batch_size x mem_dim]

        """


        # combine the new read vector with the prior memory state (w1)
        mi_info = self.concat_layer(torch.cat([read_vector, memory_state], 1))
        next_memory_state = mi_info  # new memory state if no self-attention & memory-gating



        return next_memory_state
