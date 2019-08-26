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
summary_unit.py: Implementation of the ``SummaryUpdateUnit`` for the VWM network.
"""
__author__ = "Vincent Albouy, T.S. Jayram"

import torch
from torch.nn import Module
from miprometheus.models.vwm_model.utils_VWM import linear


class SummaryUpdateUnit(Module):
    """
    Implementation of the ``SummaryUpdateUnit`` of the MAC network.
    """

    def __init__(self, dim):
        """
        Constructor for the ``SummaryUpdateUnit``.

        :param dim: dimension of feature vectors
        :type dim: int

        """

        # call base constructor
        super(SummaryUpdateUnit, self).__init__()

        # linear layer for the concatenation of context_output and summary_output
        self.concat_layer = linear(2 * dim, dim, bias=True)

    def forward(self, image_match, visual_object, memory_match, memory_object, summary_object):
        """
        Forward pass of the ``SummaryUpdateUnit``.

        :param image_match
        :param visual_object
        :param memory_match
        :param memory_object
        :param summary_object

        :return: new_summary_object
        """

        # compute new relevant object
        relevant_object = (image_match[..., None] * visual_object
                           + memory_match[..., None] * memory_object)

        # combine the new read vector with the prior memory state (w1)
        new_summary_object = self.concat_layer(
            torch.cat([relevant_object, summary_object], dim=1))

        return new_summary_object
