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
visual_retrieval.py: Implementation of the ``VisualRetrievalUnitt`` for the VWM network.
"""

__author__ = "Vincent Albouy, T.S. Jayram"

from torch.nn import Module

from miprometheus.models.vwm_model.interaction_module import InteractionModule
from miprometheus.models.vwm_model.attention_module import AttentionModule


class VisualRetrievalUnit(Module):
    """
    Implementation of the ``VisualReadUnit`` of the VWM network.
    """

    def __init__(self, dim):
        """
        Constructor for the ``VisualReadUnit``.

        :param dim: dimension of feature vectors
        :type dim: int

        """

        # call base constructor
        super(VisualRetrievalUnit, self).__init__()

        # instantiate interaction module
        self.interaction_module = InteractionModule(dim, do_project=False)

        # instantiate attention module
        self.attention_module = AttentionModule(dim)

    def forward(self, summary_object, feature_maps, feature_maps_proj, control_state):
        """
        Forward pass of the ``VisualRetrievalUnit``. Assuming 1 scalar attention weight per \
        knowledge base elements.

        :param summary_object:  previous summary object [batch_size x dim]
        :param feature_maps: image representation (output of CNN)  [batch_size x (H*W) x dim]
        :param feature_maps_proj: [batch_size x num_objects x dim]
        :param control_state:  previous control state [batch_size x dim].

        :return: visual_object [batch_size x dim]
        :return: visual_attention [batch_size x (H*W)]

        """
        feature_maps_modified = self.interaction_module(
            summary_object, feature_maps, feature_maps_proj)

        # compute attention weights
        visual_object, visual_attention = self.attention_module(
            control_state, feature_maps_modified, feature_maps)

        return visual_object, visual_attention
