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
        Constructor for the ``InteractionModule``.
        :param dim: global 'd' hidden dimension
        :type dim: int
        """

        # call base constructor
        super(InteractionModule, self).__init__()

        # linear layer for the projection of the query
        self.base_object_proj_layer = linear(dim, dim, bias=True)

        # linear layer for the projection of the keys
        self.feature_objects_proj_layer = linear(dim, dim, bias=True)

        # linear layer to define I'(i,h,w) elements
        self.modifier = linear(2 * dim, dim, bias=True)

    def forward(self, base_object, feature_objects):
        """
        Forward pass of the ``InteractionModule``.

        :param base_object: query [batch_size x dim]
        :type base_object: torch.tensor

        :param  feature_objects: [batch_size x num_objects x dim]
        :type feature_objects: torch.tensor

        :return: feature_objects_modified [batch_size x num_objects x dim]
        """

        # pass query object through linear layer
        base_object_proj = self.base_object_proj_layer(base_object)
        # [batch_size x dim]
        print(f'Base shape: {base_object_proj.size()}')

        # pass feature_objects through linear layer
        feature_objects_proj = self.feature_objects_proj_layer(feature_objects)
        # [batch_size x num_objects x dim]
        print(f'Feature shape: {feature_objects_proj.size()}')

        # modify the projected feature objects using the projected base object
        feature_objects_modified = torch.cat([
            base_object_proj[:, None, :] * feature_objects_proj,
            feature_objects], dim=-1)
        feature_objects_modified = self.modifier(feature_objects_modified)
        # [batch_size x num_objects x dim]

        return feature_objects_modified
