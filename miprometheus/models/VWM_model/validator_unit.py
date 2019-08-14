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
validator_unit.py: Implementation of the ``ValidatorUnit`` for the VWM network.

"""
__author__ = "Vincent Albouy, T.S. Jayram"

import torch
from torch.nn import Module
from miprometheus.models.VWM_model.utils_VWM import linear


class ValidatorUnit(Module):
    """
    Implementation of the `` ValidatorUnit`` of the VWM network.
    """

    def __init__(self, dim):
        """
        Constructor for the `` ValidatorUnit``.
        :param dim: dimension of feature vectors
        :type dim: int
        """

        # call base constructor
        super(ValidatorUnit, self).__init__()

        def two_layers_net():
            return torch.nn.Sequential(linear(2 * dim, 2 * dim, bias=True),
                                       torch.nn.ELU(),
                                       linear(2 * dim, 1, bias=True),
                                       torch.nn.Sigmoid())

        self.visual_object_validator = two_layers_net()
        self.memory_object_validator = two_layers_net()

    def forward(self, control_state, visual_object, memory_object, temporal_class_weights):
        """
        Forward pass of the ``ValidatorUnit``.

        :param control_state: last control state
        :param visual_object: visual output
        :param memory_object: memory output
        :param temporal_class_weights

        :return: is_visual, is_mem, do_replace, do_add_new
        """

        # the visual object validator
        concat_read_visual = torch.cat([control_state, visual_object], dim=1)
        valid_vo = self.visual_object_validator(concat_read_visual)
        valid_vo = valid_vo.squeeze(-1)

        # the memory object validator
        concat_read_memory = torch.cat([control_state, memory_object], dim=1)
        valid_mo = self.memory_object_validator(concat_read_memory)
        valid_mo = valid_mo.squeeze(-1)

        from miprometheus.models.VWM_model.utils_VWM import eval_predicate
        do_replace, do_add_new, is_visual, is_mem = eval_predicate(
            temporal_class_weights, valid_vo, valid_mo)

        return valid_vo, valid_mo, do_replace, do_add_new, is_visual, is_mem
