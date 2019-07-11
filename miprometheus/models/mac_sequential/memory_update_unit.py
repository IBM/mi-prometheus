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
read_unit.py: Implementation of the ``MemoryRetrievalUnit`` for the VWM network. Cf https://arxiv.org/abs/1803.03067 for the \
reference paper.
"""
__author__ = "Vincent Albouy"

import torch
from torch.nn import Module
from miprometheus.utils.app_state import AppState
app_state = AppState()



class MemoryUpdateUnit(Module):
    """
    Implementation of the `` MemoryUpdateUnit`` of the VWM network.
    """

    def __init__(self,dim,slots):
        """
        Constructor for the `` MemoryUpdateUnit``.
        :param dim: global 'd' hidden dimension
        :type dim: int
        """

        # call base constructor
        super(MemoryUpdateUnit, self).__init__()

        self.dim=dim
        self.slots=slots


        if slots==4:
            self.convolution_kernel = torch.tensor(
                [[0., 0., 0., 1.], [1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]).type(app_state.dtype)

        elif slots==6:
            self.convolution_kernel = torch.tensor(
                [[0., 0., 0., 0., 0., 1.], [1., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0.],
                 [0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 1., 0.]]).type(app_state.dtype)

        elif slots==8:
            self.convolution_kernel = torch.tensor(
                [[0., 0., 0., 0., 0., 0., 0., 1.], [1., 0., 0., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 1., 0., 0., 0., 0., 0.], [0., 0., 0., 1., 0., 0., 0., 0.], [0., 0., 0., 0., 1., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 0., 0., 1., 0.]]).type(app_state.dtype)


        elif slots == 10:
            self.convolution_kernel = torch.tensor(
                [[0., 0., 0., 0., 0., 0., 0.,0.,0., 1.], [1., 0.,0.,0., 0., 0., 0., 0., 0., 0.], [0., 1., 0.,0.,0., 0., 0., 0., 0., 0.],
                 [0., 0., 1., 0.,0.,0., 0., 0., 0., 0.], [0., 0., 0., 1., 0., 0., 0.,0.,0., 0.], [0., 0., 0., 0., 1., 0., 0.,0.,0., 0.],
                 [0., 0., 0., 0., 0., 1., 0.,0.,0., 0.], [0., 0., 0., 0., 0., 0., 1.,0.,0., 0.], [0., 0., 0., 0., 0., 0., 0.,1.,0., 0.],[0., 0., 0., 0., 0., 0., 0.,0.,1., 0.]]).type(app_state.dtype)

        else:
            exit()



    def forward(self, gkb, gmem, read, read_history,rvi_history, visual_working_memory,  context_weighting_vector_T,  Wt_sequential ):
        """
        Forward pass of the ``MemoryRetrievalUnit``. Assuming 1 scalar attention weight per \
        knowledge base elements.
        :param memory_states: list of all previous memory states, each of shape [batch_size x mem_dim]
        :type memory_states: torch.tensor
        :param  history: image representation (output of CNN), shape [batch_size x nb_kernels x (feat_H * feat_W)]
        :type history: torch.tensor
        :param ctrl_states: All previous control state, each of shape [batch_size x ctrl_dim].
        :type ctrl_states: list
        :return: gkb, gmem
        """
        # calculate two gates gKB and gM gates

        batch_size=read.size(0)


        # choose between now, last, latest context to built the final read vector
        now_context = gkb * read
        last_context = gmem * read_history
        latest_context = (1 - gkb) * last_context + now_context

        context_weighting_vector_T = context_weighting_vector_T.unsqueeze(1)
        T1 = context_weighting_vector_T[:, :, 0]
        T2 = context_weighting_vector_T[:, :, 1]
        T3 = context_weighting_vector_T[:, :, 2]
        T4 = context_weighting_vector_T[:, :, 3]

        # obtain alpha and beta
        alpha = gmem * gkb * (T2 + T3) * (1 - T4)
        beta = (1 - gmem) * gkb * (T2 + T3) * (1 - T4)

        # history update equation
        W = alpha * rvi_history + Wt_sequential.squeeze(1) * beta
        W = W.unsqueeze(1)

        ######## Update history #########

        # take the read vector and add one dimension [batch size, hidden dim, 1]

        read_unsqueezed = read.unsqueeze(2)
        added_object = read_unsqueezed.matmul(W)

        unity_matrix = torch.ones(batch_size, self.dim, 1).type(app_state.dtype)
        J = torch.ones(batch_size, self.dim, self.slots).type(app_state.dtype)

        visual_working_memory = visual_working_memory * (J - unity_matrix.matmul(W)) + added_object

        ####### Update Wt_sequential ########

        # get convolved tensor

        convolved_Wt_sequential = Wt_sequential.squeeze(1).matmul(self.convolution_kernel).unsqueeze(1)

        # final expression to update Wt_sequential

        Wt_sequential = (convolved_Wt_sequential.squeeze(1) * beta).unsqueeze(1) + (
        Wt_sequential.squeeze(1) * (1 - beta)).unsqueeze(1)

        # final read vector
        context_read_vector = T1 * now_context + T2 * last_context + T3 * latest_context


        return context_read_vector