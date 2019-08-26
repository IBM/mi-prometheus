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
vwm_cell.py: Implementation of the VWM Cell for the VWM network.
"""
__author__ = "Vincent Albouy, T.S. Jayram"

from torch.nn import Module

from miprometheus.models.vwm_model.visual_retrieval_unit import VisualRetrievalUnit
from miprometheus.models.vwm_model.summary_unit import SummaryUpdateUnit
from miprometheus.models.vwm_model.memory_retrieval_unit import MemoryRetrievalUnit
from miprometheus.models.vwm_model.reasoning_unit import ReasoningUnit
from miprometheus.utils.app_state import AppState


class VWMCell(Module):
    """
    Implementation of the ``VWM Cell`` (iteration over the MAC cell) of the MAC network.
    """

    def __init__(self, dim):
        """
        Constructor for the ``VWM Cell``, which represents the recurrent cell of vwm_model.

        :param dim: dimension of feature vectors
        :type dim: int

        """

        # call base constructor
        super(VWMCell, self).__init__()

        # instantiate all the units within VWM_cell
        self.visual_retrieval_unit = VisualRetrievalUnit(dim)
        self.summary_unit = SummaryUpdateUnit(dim)

    def forward(self, summary_object, control_all,
                feature_maps, feature_maps_proj):

        """
        Forward pass of the ``VWMCell`` of VWM network

        :param summary_object:  recurrent [batch_size x dim]

        :param control_all: tuple of all info from the question driven controller
        and equals (control_state, control_attention)
        :type control_all: tuple

        :param feature_maps: feature maps (feature maps extracted by a CNN)
        [batch_size x nb_kernels x (feat_H * feat_W)].

        :param feature_maps_proj: linear projection of feature maps
        [batch_size x nb_kernels x (feat_H * feat_W)].

        :return: new_summary_object
        :return: (visual_object, read_head, do_replace, do_add_new)

        """

        control_state, control_attention = control_all

        # visual retrieval unit, obtain visual output and visual attention
        visual_object, visual_attention = self.visual_retrieval_unit(
            summary_object, feature_maps, feature_maps_proj, control_state)

        # summary update Unit
        new_summary_object = self.summary_unit(summary_object, visual_object)

        # package all the vwm cell state info
        vwm_cell_info = dict(
            cs=control_state, ca=control_attention,
            vo=visual_object, va=visual_attention)

        return new_summary_object, vwm_cell_info
