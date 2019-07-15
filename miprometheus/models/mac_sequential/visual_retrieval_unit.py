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
read_unit.py: Implementation of the ``VisualRetrievalUnitt`` for the VWM network. 
"""
__author__ = "Vincent Albouy, T.S. Jayram"

import torch
from torch.nn import Module

from miprometheus.models.mac_sequential.interaction_module import InteractionModule
from miprometheus.models.mac_sequential.attention_module import AttentionModule


class VisualRetrievalUnit(Module):
    """
    Implementation of the ``VisualReadUnit`` of the VWM network.
    """

    def __init__(self, dim):
        """
        Constructor for the ``VisualReadUnit``.

        :param dim: global 'd' hidden dimension
        :type dim: int

        """

        # call base constructor
        super(VisualRetrievalUnit, self).__init__()

        # instantiate interaction module
        self.interaction_module = InteractionModule(dim)

        # instantiate attention module
        self.attention_module = AttentionModule(dim)

    def forward(self, summary_object, feature_maps, ctrl_state):
        """
        Forward pass of the ``VisualRetrievalUnit``. Assuming 1 scalar attention weight per \
        knowledge base elements.

        :param summary_object:  previous summary object [batch_size x dim]
        :type summary_object: torch.tensor

        :param feature_maps: image representation (output of CNN) \
               batch_size x (H*W) x dim
        :type feature_maps: torch.tensor

        :param ctrl_state:  previous control state [batch_size x dim].
        :type ctrl_state: torch.tensor

        :return: visual_output [batch_size x dim]
        :return: visual_attention [batch_size x (H*W)]

        """

        feature_maps_modified = self.interaction_module(summary_object, feature_maps)
        # compute attention weights
        visual_output, visual_attention = \
            self.attention_module(ctrl_state, feature_maps_modified, feature_maps)

        return visual_output, visual_attention
