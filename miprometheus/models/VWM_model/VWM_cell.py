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
VWM_cell.py: Implementation of the VWM Cell for the VWM network. 
"""
__author__ = "Vincent Albouy, T.S. Jayram"

import torch
from torch.nn import Module

from miprometheus.models.VWM_model.question_driven_controller import QuestionDrivenController
from miprometheus.models.VWM_model.visual_retrieval_unit import VisualRetrievalUnit
from miprometheus.models.VWM_model.summary_unit import SummaryUpdateUnit
from miprometheus.models.VWM_model.memory_retrieval_unit import MemoryRetrievalUnit
from miprometheus.models.VWM_model.reasoning_unit import ReasoningUnit
from miprometheus.models.VWM_model.memory_update_unit import memory_update
from miprometheus.utils.app_state import AppState
app_state = AppState()


class VWMCell(Module):
    """
    Implementation of the ``VWM Cell`` (iteration over the MAC cell) of the MAC network.
    """

    def __init__(self, dim, max_step=12):
        """
        Constructor for the ``VWM Cell``, which represents the recurrent cell of VWM_model.

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

        # initialize hidden states
        self.mem_0 = torch.nn.Parameter(torch.zeros(1, dim).type(app_state.dtype))
        self.control_0 = torch.nn.Parameter(torch.zeros(1, dim).type(app_state.dtype))

    def forward(self, step, contextual_words, question_encoding,
                feature_maps, control_state, summary_object,
                visual_working_memory, wt_sequential, cell_state_history):

        """
        Forward pass of the ``VWMCell`` of VWM network

        :param step: step
        :type step: int

        :param contextual_words: contextual words, shape [batch_size x maxQuestionLength x dim]
        :type contextual_words: torch.tensor

        :param question_encoding: questions encodings, shape [batch_size x (2*dim)]
        :type question_encoding: torch.tensor

        :param feature_maps: feature maps (feature maps extracted by a CNN), shape \
        [batch_size x nb_kernels x (feat_H * feat_W)].
        :type feature_maps: torch.tensor
        
        :param control_state: control_state state
        :type control_state: torch.tensor
        
        :param summary_object: summary_object
        :type summary_object: torch.tensor
        
        :param visual_working_memory: visual_working_memory
        :type visual_working_memory: torch.tensor
           
        :param wt_sequential: wt_sequential
        :type wt_sequential: torch.tensor

        :param cell_state_history: [[note: modified in the method]]
        :type cell_state_history: list


        :return summary_object, shape [batch_size x dim]
        :type summary_object: torch.tensor

        :return: control_state: shape [batch_size x dim]
        :type  control_state: torch.tensor
        
        :return: visual_attention: shape [batch_size x HxW]
        :type:  visual_attention: torch.tensor
        
        :return: visual_working_memory: shape [batch_size x slots x dim]
        :type  visual_working_memory: torch.tensor
               
        :return: wt_sequential: shape [batch_size x slots]
        :type  wt_sequential: torch.tensor

        """

        # control_state unit
        (control_state, control_attention,
         temporal_class_weights) = self.question_driven_controller(
            step, contextual_words, question_encoding, control_state)

        # visual retrieval unit, obtain visual output and visual attention
        visual_object, visual_attention = self.visual_retrieval_unit(
            summary_object, feature_maps, control_state)

        # memory retrieval unit, obtain memory output and memory attention
        memory_object, memory_attention = self.memory_retrieval_unit(
            summary_object, visual_working_memory, control_state)

        # reason about the objects
        do_replace, do_add_new, is_visual, is_mem = self.reasoning_unit(
            control_state, visual_object, memory_object, temporal_class_weights)

        # update visual_working_memory, and wt sequential
        visual_working_memory, wt_sequential = memory_update(
            visual_object, memory_attention, visual_working_memory,
            wt_sequential, do_replace, do_add_new)

        # summary update Unit
        summary_object = self.summary_unit(
            is_visual, visual_object, is_mem, memory_object, summary_object)

        # store attention weights for visualization
        if app_state.visualize:
            cell_state_history.append(
                (visual_attention.detach(), control_attention.detach(),
                 visual_working_memory.detach(), memory_attention.detach(),
                 is_visual.detach().numpy(), is_mem.detach().numpy(),
                 wt_sequential.unsqueeze(1).detach().numpy(),
                 temporal_class_weights.unsqueeze(1).detach().numpy()))

        return (summary_object, control_state, visual_attention, visual_working_memory,
                wt_sequential)
