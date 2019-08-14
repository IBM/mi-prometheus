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
from miprometheus.models.VWM_model.utils_VWM import linear


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
                                       torch.nn.ELU(),
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

        do_replace, do_add_new, is_visual, is_mem = ReasoningUnit.eval_predicate(
            temporal_class_weights, valid_vo, valid_mo)

        return valid_vo, valid_mo, do_replace, do_add_new, is_visual, is_mem

    @staticmethod
    def eval_predicate(temporal_class_weights, valid_vo, valid_mo):

        # get t1,t2,t3,t4 from temporal_class_weights
        # corresponds to now, last, latest, or none
        t1 = temporal_class_weights[:, 0]
        t2 = temporal_class_weights[:, 1]
        t3 = temporal_class_weights[:, 2]
        t4 = temporal_class_weights[:, 3]

        # if the temporal context last or latest,
        # then do we replace the existing memory object?
        do_replace = valid_mo * valid_vo * (t2 + t3) * (1 - t4)

        # otherwise do we add a new one to memory?
        do_add_new = (1 - valid_mo) * valid_vo * (t2 + t3) * (1 - t4)

        # (now or latest) and valid visual object?
        is_visual = (t1 + t3) * valid_vo
        # optional extra check that it is neither last nor none
        # is_visual = is_visual * (1 - t2) * (1 - t4)

        # (now or (latest and (not valid visual object))) and valid memory object?
        is_mem = (t2 + t3 * (1 - valid_vo)) * valid_mo
        # optional extra check that it is neither now nor none
        # is_mem = is_mem * (1 - t1) * (1 - t4)

        return do_replace, do_add_new, is_visual, is_mem
