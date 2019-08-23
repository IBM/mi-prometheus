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

import torch
from torch.nn import Module

from miprometheus.models.vwm_model.question_driven_controller import QuestionDrivenController
from miprometheus.models.vwm_model.visual_retrieval_unit import VisualRetrievalUnit
from miprometheus.models.vwm_model.summary_unit import SummaryUpdateUnit
from miprometheus.models.vwm_model.memory_retrieval_unit import MemoryRetrievalUnit
from miprometheus.models.vwm_model.reasoning_unit import ReasoningUnit
from miprometheus.models.vwm_model.memory_update_unit import memory_update
from miprometheus.utils.app_state import AppState
app_state = AppState()


class VWMCell(Module):
    """
    Implementation of the ``VWM Cell`` (iteration over the MAC cell) of the MAC network.
    """

    def __init__(self, dim, max_step=12):
        """
        Constructor for the ``VWM Cell``, which represents the recurrent cell of vwm_model.

        :param dim: dimension of feature vectors
        :type dim: int

        :param max_step: maximal number of MAC cells. Default: 12
        :type max_step: int

        """

        # call base constructor
        super(VWMCell, self).__init__()

        # instantiate all the units within VWM_cell
        self.question_driven_controller = QuestionDrivenController(dim, max_step)
        self.visual_retrieval_unit = VisualRetrievalUnit(dim)
        self.memory_retrieval_unit = MemoryRetrievalUnit(dim)
        self.reasoning_unit = ReasoningUnit(dim)
        self.summary_unit = SummaryUpdateUnit(dim)

        self.cell_history = []
        self.mem_0 = torch.nn.Parameter(torch.zeros(1, dim))
        self.control_0 = torch.nn.Parameter(
            torch.zeros(1, dim))

    def forward(self, step, contextual_words, question_encoding, feature_maps,
                control_state, summary_object, visual_working_memory, write_head):

        """
        Forward pass of the ``VWMCell`` of VWM network

        :param step: step
        :type step: int

        :param contextual_words: contextual words  [batch_size x maxQuestionLength x dim]

        :param question_encoding: questions encodings shape [batch_size x (2*dim)]

        :param feature_maps: feature maps (feature maps extracted by a CNN)
        [batch_size x nb_kernels x (feat_H * feat_W)].

        :param control_state: recurrent  [batch_size x dim]
        :param summary_object:  recurrent [batch_size x dim]
        :param visual_working_memory: recurrent [batch_size x slots x dim]
        :param write_head: recurrent [batch_size x num_slots]

        :return: new_control_state, new_summary_object, new_visual_working_memory,
                 new_write_head

        """

        # new_control_state unit
        (new_control_state, control_attention,
         temporal_class_weights) = self.question_driven_controller(
            step, contextual_words, question_encoding, control_state)

        # visual retrieval unit, obtain visual output and visual attention
        visual_object, visual_attention = self.visual_retrieval_unit(
            summary_object, feature_maps, new_control_state)

        # memory retrieval unit, obtain memory output and memory attention
        memory_object, read_head = self.memory_retrieval_unit(
            summary_object, visual_working_memory, new_control_state)

        # reason about the objects
        image_match, memory_match, do_replace, do_add_new = self.reasoning_unit(
            new_control_state, visual_object, memory_object, temporal_class_weights)

        # update visual_working_memory, and wt sequential
        new_visual_working_memory, new_write_head = memory_update(
            visual_object, visual_working_memory, read_head, write_head,
            do_replace, do_add_new)

        # summary update Unit
        new_summary_object = self.summary_unit(
            image_match, visual_object, memory_match, memory_object, summary_object)

        # store attention weights for visualization
        if app_state.visualize:
            cell_info = [x.detach() for x in [
                visual_attention, control_attention, new_visual_working_memory,
                read_head, image_match, memory_match, new_write_head,
                temporal_class_weights]]

            cell_info.insert(0, step)

            self.cell_history.append(tuple(cell_info))

        return (new_control_state, new_summary_object,
                new_visual_working_memory, new_write_head)
