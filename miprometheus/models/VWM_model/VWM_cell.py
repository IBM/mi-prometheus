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
mac_unit.py: Implementation of the VWM Cell for the VWM network. 
"""
__author__ = "Vincent Albouy, T.S. Jayram"

import torch
from torch.nn import Module

from miprometheus.models.VWM_model.question_driven_controller import QuestionDrivenController
from miprometheus.models.VWM_model.visual_retrieval_unit import VisualRetrievalUnit
from miprometheus.models.VWM_model.thought_unit import ThoughtUnit
from miprometheus.models.VWM_model.memory_retrieval_unit import MemoryRetrievalUnit
from miprometheus.models.VWM_model.validator_unit import ValidatorUnit
from miprometheus.models.VWM_model.memory_update_unit import MemoryUpdateUnit
from miprometheus.utils.app_state import AppState
app_state = AppState()


class VWMCell(Module):
    """
    Implementation of the ``VWM Cell`` (iteration over the MAC cell) of the MAC network.
    """

    def __init__(self, dim, max_step=12, dropout=0.15, slots=0):
        """
        Constructor for the ``VWM Cell``, which represents the recurrence over the \
        MACCell.

        :param dim: global 'd' hidden dimension.
        :type dim: int

        :param max_step: maximal number of MAC cells. Default: 12
        :type max_step: int


        :param dropout: dropout probability for the variational dropout mask. Default: 0.15
        :type dropout: float

        """

        # call base constructor
        super(VWMCell, self).__init__()

        # instantiate the units
        self.question_driven_controller = QuestionDrivenController(dim=dim, max_step=max_step)
        self.visual_retrieval_unit = VisualRetrievalUnit(dim=dim)
        self.memory_retrieval_unit = MemoryRetrievalUnit(dim=dim)
        self.validator_unit = ValidatorUnit(dim=dim)
        self.memory_update_unit = MemoryUpdateUnit(dim=dim, slots=slots)
        self.thought_unit = ThoughtUnit(
            dim=dim)


        # initialize hidden states
        self.mem_0 = torch.nn.Parameter(torch.zeros(1, dim).type(app_state.dtype))
        self.control_0 = torch.nn.Parameter(
            torch.zeros(1, dim).type(app_state.dtype))

        #contantes values
        self.dim = dim
        self.dropout = dropout
        self.slots = slots

        #Visualization
        self.cell_state_history = []



    def forward(self, context, question, features_maps, control, summary_output, visual_working_memory, Wt_sequential,step):
        """
        Forward pass of the ``VWMCell``, which represents the recurrence over the \
        MACCell.

        :param context: contextual words, shape [batch_size x maxQuestionLength x dim]
        :type context: torch.tensor

        :param question: questions encodings, shape [batch_size x 2*dim]
        :type question: torch.tensor

        :param feature maps: feature mapse (feature maps extracted by a CNN), shape \
        [batch_size x nb_kernels x (feat_H * feat_W)].
        :type feature maps: torch.tensor
        
        :param control: control state
        :type control: torch.tensor
        
        :param summary_output: summary_output
        :type summary_output: torch.tensor
        
        :param visual_working_memory: visual_working_memory
        :type visual_working_memory: torch.tensor
           
        :param Wt_sequential: Wt_sequential
        :type Wt_sequential: torch.tensor
               

        :return: summary_output, self.cell_state_history, va, visual_working_memory, Wt_sequential

        """

        # empty state history
        self.cell_state_history = []

        # control unit\
        control, control_attention, context_weighting_vector_T = self.question_driven_controller(
            step=step,
            contextual_words=context,
            question_encoding=question,
            ctrl_state=control)

        # visual retrieval unit
        #print(f'Shapes FM: {summary_output.size()}, {features_maps.size()}, {control.size()}')

        vo, va = self.visual_retrieval_unit(summary_object=summary_output, feature_maps=features_maps,
                         ctrl_state=control)

        # memory retrieval unit
        #print(f'Shapes VWM: {summary_output.size()}, {visual_working_memory.size()}, {control.size()}')
        mo, ma = self.memory_retrieval_unit(summary_object=summary_output, visual_working_memory=visual_working_memory,
                          ctrl_state=control)

        # matching unit
        gvt,gmt=self.validator_unit(control,vo,mo)

        #update visual_working_memory , wt sequential and get the final context vector
        context_output, visual_working_memory, Wt_sequential = self.memory_update_unit(gvt, gmt, vo, mo, ma, visual_working_memory,
                                        context_weighting_vector_T, Wt_sequential)

        # thought Unit
        summary_output = self.thought_unit(summary_output=summary_output,
                                            context_output= context_output)

        # store attention weights for visualization
        if app_state.visualize:
            self.cell_state_history.append(
                (va.detach(), control_attention.detach(), visual_working_memory.detach(), ma.detach(),gvt,gmt, Wt_sequential, context_weighting_vector_T))

        return summary_output, control,  self.cell_state_history, va, visual_working_memory, Wt_sequential
