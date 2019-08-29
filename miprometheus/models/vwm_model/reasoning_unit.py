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
reasoning_unit.py: Implementation of the ``ReasoningUnit`` for the VWM network.

"""
__author__ = "Vincent Albouy, T.S. Jayram"

import torch
from torch.nn import Module
from miprometheus.models.vwm_model.utils_VWM import linear


class ReasoningUnit(Module):
    """
    Implementation of the `` ReasoningUnit`` of the VWM network.
    """

    def __init__(self, dim):
        """
        Constructor for the `` ReasoningUnit``.
        :param dim: dimension of feature vectors
        :type dim: int
        """

        # call base constructor
        super(ReasoningUnit, self).__init__()

        def two_layers_net():
            return torch.nn.Sequential(linear(2 * dim, 2 * dim, bias=True),
                                       torch.nn.ReLU(),
                                       linear(2 * dim, 1, bias=True),
                                       torch.nn.Sigmoid())

        self.visual_object_validator = two_layers_net()
        self.memory_object_validator = two_layers_net()

    def forward(self, control_state, visual_object, memory_object, temporal_class_weights):
        """
        Forward pass of the ``ReasoningUnit``.

        :param control_state: last control state
        :param visual_object: visual output
        :param memory_object: memory output
        :param temporal_class_weights

        :return: image_match, memory_match, do_replace, do_add_new
        """

        # the visual object validator
        concat_read_visual = torch.cat([control_state, visual_object], dim=1)
        valid_vo = self.visual_object_validator(concat_read_visual)
        valid_vo = valid_vo.squeeze(-1)

        # the memory object validator
        concat_read_memory = torch.cat([control_state, memory_object], dim=1)
        valid_mo = self.memory_object_validator(concat_read_memory)
        valid_mo = valid_mo.squeeze(-1)

        # get t_now, t_last, t_latest, t_none from temporal_class_weights
        t_now = temporal_class_weights[:, 0]
        t_last = temporal_class_weights[:, 1]
        t_latest = temporal_class_weights[:, 2]
        t_none = temporal_class_weights[:, 3]

        # check if temporal context is last or latest
        # temporal_test_1 = torch.relu(t_last + t_latest - t_now - t_none)
        temporal_test_1 = (t_last + t_latest) * (1 - t_now)

        # conditioned on temporal context,
        # check if we should replace existing memory object
        do_replace = valid_mo * valid_vo * temporal_test_1

        # otherwise, conditioned on temporal context,
        # check if we should add a new one to VWM
        do_add_new = (1 - valid_mo) * valid_vo * temporal_test_1

        # check if temporal context is now or latest
        temporal_test_2 = (t_now + t_latest) * (1 - t_last)

        # conditioned on temporal context, check if we have a valid visual object
        image_match = valid_vo * temporal_test_2

        # check if temporal context is either last, or latest without a visual object
        temporal_test_3 = (t_last + t_latest * (1 - valid_vo)) * (1 - t_now)

        # conditioned on temporal context, check if we have a valid memory object
        memory_match = valid_mo * temporal_test_3

        return image_match, memory_match, do_replace, do_add_new
