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
interaction_module.py: Implementation of the ``Interaction Module`` for the VWM network.
"""
__author__ = "Vincent Albouy, T.S. Jayram"

import torch
from torch.nn import Module

from miprometheus.models.vwm_model.utils_VWM import linear


class InteractionModule(Module):
    """
    Implementation of the ``Interaction Unit`` of the VWM network.
    """

    def __init__(self, dim, do_project=True):
        """
        Constructor for the ``InteractionModule``.
        :param dim: dimension of feature vectors
        :type dim: int

        :param do_project: flag indicating whether we need to project the keys
        :type do_project: Boolean
        """

        # call base constructor
        super(InteractionModule, self).__init__()

        # linear layer for the projection of the query
        self.base_object_proj_layer = linear(dim, dim, bias=True)

        # linear layer for the projection of the keys
        if do_project:
            self.feature_objects_proj_layer = linear(dim, dim, bias=True)

        self.do_project = do_project

        # linear layer to define the modulation of feature objects by base object
        self.modifier = linear(2 * dim, dim, bias=True)

    def forward(self, base_object, feature_objects, feature_objects_proj=None):
        """
        Forward pass of the ``InteractionModule``.

        :param base_object: query [batch_size x dim]
        :param  feature_objects: [batch_size x num_objects x dim]
        :param  feature_objects_proj: [batch_size x num_objects x dim]

        :return: feature_objects_modified [batch_size x num_objects x dim]
        """

        if self.do_project:
            assert feature_objects_proj is None
        else:
            assert feature_objects.size() == feature_objects_proj.size(), (
                'Shape mismatch between feature_objects and feature_objects_proj')

        # pass query object through linear layer
        base_object_proj = self.base_object_proj_layer(base_object)
        # [batch_size x dim]

        # pass feature_objects through linear layer
        if self.do_project:
            feature_objects_proj = self.feature_objects_proj_layer(feature_objects)
        # [batch_size x num_objects x dim]

        # modify the projected feature objects using the projected base object
        feature_objects_modified = torch.cat([
            base_object_proj[:, None, :] * feature_objects_proj,
            feature_objects], dim=-1)
        feature_objects_modified = self.modifier(feature_objects_modified)
        # [batch_size x num_objects x dim]

        return feature_objects_modified
