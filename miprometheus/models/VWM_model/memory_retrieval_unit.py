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
memory_retrieval.py: Implementation of the ``MemoryRetrievalUnit`` for the VWM network.
"""
__author__ = "Vincent Albouy, T.S. Jayram"

from torch.nn import Module

from miprometheus.models.VWM_model.attention_module import AttentionModule
from miprometheus.models.VWM_model.interaction_module import InteractionModule


class MemoryRetrievalUnit(Module):
    """
    Implementation of the ``MemoryRetrievalUnit`` of the VWM network.
    """

    def __init__(self, dim):
        """
        Constructor for the ``MemoryRetrievalUnit``.
        :param dim: dimension of feature vectors
        :type dim: int
        """

        # call base constructor
        super(MemoryRetrievalUnit, self).__init__()

        # instantiate interaction module
        self.interaction_module = InteractionModule(dim)

        # instantiate attention module
        self.attention_module = AttentionModule(dim)

    def forward(self, summary_object, visual_working_memory, control_state):
        """
        Forward pass of the ``MemoryRetrievalUnit``. Assuming 1 scalar attention weight per \
        knowledge base elements.
        
        :param summary_object:  previous summary object [batch_size x dim]
        :param visual_working_memory: batch_size x vwm_num_slots x dim
        :param control_state:  previous control state [batch_size x dim].

        :return: memory_object [batch_size x dim]
        :return: read_head [batch_size x vwm_num_slots]
        """

        # Combine the summary object with VWM
        vwm_modified = self.interaction_module(summary_object, visual_working_memory)

        # compute attention weights
        memory_object, read_head = self.attention_module(
            control_state, vwm_modified, visual_working_memory)

        return memory_object, read_head
