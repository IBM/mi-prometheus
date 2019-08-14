#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2018 Kim Seonghyeon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ------------------------------------------------------------------------------
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
from miprometheus.models.VWM_model.memory_update_unit import MemoryUpdateUnit
from miprometheus.utils.app_state import AppState
app_state = AppState()


class VWMCell(Module):
    """
    Implementation of the ``VWM Cell`` (iteration over the MAC cell) of the MAC network.
    """

    def __init__(self, dim, max_step=12, dropout=0.15, slots=0):
        """
        Constructor for the ``VWM Cell``, which represents the recurrent cell of VWM_model.

        :param dim: global 'd' hidden dimension.
        :type dim: int

        :param max_step: maximal number of MAC cells. Default: 12
        :type max_step: int

        :param dropout: dropout probability for the variational dropout mask. Default: 0.15
        :type dropout: float

        """

        # call base constructor
        super(VWMCell, self).__init__()

        # instantiate all the units within VWM_cell
        self.question_driven_controller = QuestionDrivenController(dim=dim, max_step=max_step)
        self.visual_retrieval_unit = VisualRetrievalUnit(dim=dim)
        self.memory_retrieval_unit = MemoryRetrievalUnit(dim=dim)
        self.reasoning_unit = ReasoningUnit(dim=dim)
        self.memory_update_unit = MemoryUpdateUnit(dim=dim, slots=slots)
        self.summary_unit = SummaryUpdateUnit(
            dim=dim)


        # initialize hidden states
        self.mem_0 = torch.nn.Parameter(torch.zeros(1, dim).type(app_state.dtype))
        self.control_0 = torch.nn.Parameter(
            torch.zeros(1, dim).type(app_state.dtype))

        #contantes values
        self.dim = dim
        self.dropout = dropout

        #Visualization container
        self.cell_state_history = []

    def forward(self, context, question, features_maps, control,
                summary_object, visual_working_memory, wt_sequential, state_history, step):

        """
        Forward pass of the ``VWMCell`` of VWM network

        :param context: contextual words, shape [batch_size x maxQuestionLength x dim]
        :type context: torch.tensor

        :param question: questions encodings, shape [batch_size x 2*dim]
        :type question: torch.tensor

        :param feature maps: feature maps (feature maps extracted by a CNN), shape \
        [batch_size x nb_kernels x (feat_H * feat_W)].
        :type feature maps: torch.tensor
        
        :param control: control state
        :type control: torch.tensor
        
        :param summary_object: summary_object
        :type summary_object: torch.tensor
        
        :param visual_working_memory: visual_working_memory
        :type visual_working_memory: torch.tensor
           
        :param wt_sequential: wt_sequential
        :type wt_sequential: torch.tensor
        
        :param step: step 
        :type step: int
               
        :return summary_object, shape [batch_size x dim]
        :type summary_object: torch.tensor
            
        :return self.cell_state_history
        :type self.cell_state_history: list
        
        :return: control: shape [batch_size x dim]
        :type  control: torch.tensor
        
        :return: va: shape [batch_size x HxW]
        :type:  va: torch.tensor
        
        :return: visual_working_memory: shape [batch_size x slots x dim]
        :type  visual_working_memory: torch.tensor
               
        :return: wt_sequential: shape [batch_size x slots]
        :type  wt_sequential: torch.tensor

        """

        # control unit\
        control, control_attention, context_weighting_vector_T = self.question_driven_controller(
            step=step,
            contextual_words=context,
            question_encoding=question,
            control_state=control)

        # visual retrieval unit, obtain visual output and visual attention
        vo, va = self.visual_retrieval_unit(summary_object=summary_object, feature_maps=features_maps,
                                            control_state=control)


        # memory retrieval unit, obtain memory output and memory attention
        mo, ma = self.memory_retrieval_unit(summary_object=summary_object, visual_working_memory=visual_working_memory,
                                            control_state=control)

        # matching unit
        valid_vo, valid_mo, do_replace, do_add_new, is_visual, is_mem = self.reasoning_unit(
            control,vo,mo,context_weighting_vector_T)

        # update visual_working_memory, wt sequential and get the final context output vector
        (relevant_object, visual_working_memory,
         wt_sequential, is_visual, is_mem) = self.memory_update_unit(
            valid_vo, valid_mo, vo, mo, ma, visual_working_memory,
            context_weighting_vector_T, wt_sequential,
            do_replace, is_visual, is_mem)


        # summary update Unit
        summary_object = self.summary_unit(
            is_visual, vo, is_mem, mo, summary_object)

        # summary_object = self.summary_unit(summary_object=summary_object,
        #                                     context_output= context_output)

        # store attention weights for visualization
        if app_state.visualize:
            state_history.append(
                (va.detach(), control_attention.detach(), visual_working_memory.detach(), ma.detach(), gvt.detach().numpy(), gmt.detach().numpy(), wt_sequential.unsqueeze(1).detach().numpy(), context_weighting_vector_T.unsqueeze(1).detach().numpy()))

        return (summary_object, control, state_history, va,
                visual_working_memory, wt_sequential)
