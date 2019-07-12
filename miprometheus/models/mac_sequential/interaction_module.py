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
interaction_module.py: Implementation of the ``Interaction Module`` for the VWM network.
"""
__author__ = "Vincent Albouy, T.S. Jayram"

import torch
from torch.nn import Module

from miprometheus.models.mac_sequential.utils_VWM import linear


class InteractionModule(Module):
    """
    Implementation of the ``Interaction Unit`` of the VWM network.
    """

    def __init__(self, dim):
        """
        Constructor for the ``MemoryRetrievalUnit``.
        :param dim: global 'd' hidden dimension
        :type dim: int
        """

        # call base constructor
        super(InteractionModule, self).__init__()

        # linear layer for the projection of the query
        self.query_proj_layer = linear(dim, dim, bias=True)

        # linear layer for the projection of the keys
        self.keys_proj_layer = linear(dim, dim, bias=True)

        # linear layer to define I'(i,h,w) elements
        self.modifier = linear(2 * dim, dim, bias=True)

    def forward(self, query_object, feature_objects):
        """
        Forward pass of the ``MemoryRetrievalUnit``. Assuming 1 scalar attention weight per \
        knowledge base elements.
        
        :param query_object: query [batch_size x dim]
        :type query_object: torch.tensor

        :param  feature_objects: [batch_size x dim x (H*W)]
        :type feature_objects: torch.tensor

        :return: aggregate_objects [batch_size x dim x (H*W)]
        """

        # pass summary object through linear layer
        query_object_proj = self.summary_proj_layer(query_object)  # [batch_size x dim x 1]

        # pass VWM through linear layer
        feature_objects_proj = self.feature_objects(feature_objects)

        # compute I(i,h,w) elements
        # [batch_size x dim] * [batch_size x dim x (H*W)] -> [batch_size x dim x (H*W)]
        I_elements = query_object_proj[:, None, :] * feature_objects_proj

        # compute I' elements
        modified_feature_objects = self.modifier(torch.cat(
            [I_elements, feature_objects], dim=-1))  # [batch_size x dim x (H*W)]

        return modified_feature_objects
