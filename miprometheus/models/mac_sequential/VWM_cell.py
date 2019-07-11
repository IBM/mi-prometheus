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
mac_unit.py: Implementation of the MAC Unit for the MAC network. Cf https://arxiv.org/abs/1803.03067 for the \
reference paper.
"""
__author__ = "Vincent Albouy"

import torch
from torch.nn import Module

from miprometheus.models.mac_sequential.question_driven_controller import QuestionDrivenController
from miprometheus.models.mac_sequential.visual_retrieval_unit import VisualRetrievalUnit
from miprometheus.models.mac_sequential.thought_unit import ThoughtUnit
from miprometheus.models.mac_sequential.memory_retrieval_unit import MemoryRetrievalUnit
from miprometheus.models.mac_sequential.matching_unit import MatchingUnit
from miprometheus.models.mac_sequential.memory_update_unit import MemoryUpdateUnit
from miprometheus.models.mac_sequential.utils_mac import linear
from miprometheus.utils.app_state import AppState
app_state = AppState()


class VWMCell(Module):
    """
    Implementation of the ``MACUnit`` (iteration over the MAC cell) of the MAC network.
    """

    def __init__(self, dim, max_step=2, self_attention=False,
                 memory_gate=False, dropout=0.15, slots=0):
        """
        Constructor for the ``MACUnit``, which represents the recurrence over the \
        MACCell.

        :param dim: global 'd' hidden dimension.
        :type dim: int

        :param max_step: maximal number of MAC cells. Default: 12
        :type max_step: int

        :param self_attention: whether or not to use self-attention in the ``WriteUnit``. Default: ``False``.
        :type self_attention: bool

        :param memory_gate: whether or not to use memory gating in the ``WriteUnit``. Default: ``False``.
        :type memory_gate: bool

        :param dropout: dropout probability for the variational dropout mask. Default: 0.15
        :type dropout: float

        """

        # call base constructor
        super(VWMCell, self).__init__()

        # instantiate the units
        self.question_driven_controller = QuestionDrivenController(dim=dim, max_step=max_step)
        self.visual_retrieval_unit = VisualRetrievalUnit(dim=dim)
        self.memory_retrieval_unit = MemoryRetrievalUnit(dim=dim)
        self.matching_unit = MatchingUnit(dim=dim)
        self.memory_update_unit = MemoryUpdateUnit(dim=dim, slots=slots)
        self.thought_unit = ThoughtUnit(
            dim=dim)

        self.slots=slots

        # initialize hidden states
        self.mem_0 = torch.nn.Parameter(torch.zeros(1, dim).type(app_state.dtype))
        self.control_0 = torch.nn.Parameter(
            torch.zeros(1, dim).type(app_state.dtype))

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout

        self.cell_state_history = []

        self.W = torch.zeros(48, 1, self.slots).type(app_state.dtype)


        self.linear_layer = linear(128, 1, bias=True)

        self.linear_layer_history = linear(128, 1, bias=True)

        self.concat_contexts = torch.zeros(48, 128, requires_grad=False).type(app_state.dtype)


    def get_dropout_mask(self, x, dropout):
        """
        Create a dropout mask to be applied on x.

        :param x: tensor of arbitrary shape to apply the mask on.
        :type x: torch.tensor

        :param dropout: dropout rate.
        :type dropout: float

        :return: mask.

        """
        # create a binary mask, where the probability of 1's is (1-dropout)
        mask = torch.empty_like(x).bernoulli_(
            1 - dropout).type(app_state.dtype)

        # normalize the mask so that the average value is 1 and not (1-dropout)
        mask /= (1 - dropout)

        return mask

    def forward(self, context, question, knowledge, control, summary_output, visual_working_memory,  Wt_sequential ):
        """
        Forward pass of the ``VWMCell``, which represents the recurrence over the \
        MACCell.

        :param context: contextual words, shape [batch_size x maxQuestionLength x dim]
        :type context: torch.tensor

        :param question: questions encodings, shape [batch_size x 2*dim]
        :type question: torch.tensor

        :param knowledge: knowledge_base (feature maps extracted by a CNN), shape \
        [batch_size x nb_kernels x (feat_H * feat_W)].
        :type knowledge: torch.tensor

        :return: list of the memory states.

        """

        # empty state history
        self.cell_state_history = []

        # main loop of recurrence over the MACCell
        for i in range(self.max_step):

            # control unit
            control, control_attention, context_weighting_vector_T = self.question_driven_controller(
                step=i,
                contextual_words=context,
                question_encoding=question,
                ctrl_state=control)


            # visual retrieval unit
            vo, va = self.visual_retrieval_unit(summary_object=summary_output, feature_maps=knowledge,
                             ctrl_state=control)

            # memory retrieval unit
            mo, ma = self.memory_retrieval_unit(summary_object=summary_output, visual_working_memory=visual_working_memory,
                             ctrl_state=control)

            # matching unit
            gvt,gmt=self.matching_unit(control,vo,mo)

            context_output = self.memory_update_unit(gvt, gmt, vo, mo, ma, visual_working_memory,
                                        context_weighting_vector_T, Wt_sequential)

            # thought Unit
            summary_output = self.thought_unit(summary_output=summary_output,
                                               context_output= context_output , ctrl_state=control)


            # store attention weights for visualization
            if app_state.visualize:
                self.cell_state_history.append(
                    (va.detach(), control_attention.detach(), visual_working_memory.detach(), ma.detach(),gvt,gmt, Wt_sequential, context_weighting_vector_T))

        return summary_output, self.cell_state_history, va, visual_working_memory, Wt_sequential
