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
read_unit.py: Implementation of the ``MemoryUpdateUnit`` for the VWM network.

"""

__author__ = "Vincent Albouy, T.S. Jayram"

import torch
from torch.nn import Module

from miprometheus.utils.app_state import AppState
app_state = AppState()


class MemoryUpdateUnit(Module):
    """
    Implementation of the `` MemoryUpdateUnit`` of the VWM network.
    """

    def __init__(self, dim, slots):
        """
        Constructor for the `` MemoryUpdateUnit``.
        :param dim: global 'd' hidden dimension
        :type dim: int
        """

        # call base constructor
        super(MemoryUpdateUnit, self).__init__()

        self.dim = dim

        # number of slots in memory
        self.slots = slots

    def forward(self, gvt, gmt, vo, mo, ma, visual_working_memory,
                context_weighting_vector_T, Wt_sequential):
        """
        Forward pass of the ``MemoryUpdateUnit``. 
        
        :param valid_vo : whether visual object is valid
        :param valid_mo : whether memory object is valid
        :param vo : visual object
        :param mo: memory object
        :param ma: memory attention
        :param vwm: vwm
        :param temporal_class_weights: matrix to get T1,T2,T3,T4
        :param wt_sequential :wt_sequential
        
        :return: context_read_vector, vwm, wt_sequential
        """

        batch_size = vo.size(0)

        # choose between now, last, latest context to built the final read vector

        gvt = gvt[..., None]
        gmt = gmt[..., None]

        now_context = gvt * vo
        last_context = gmt * mo
        latest_context = (1 - gvt) * last_context + now_context

        #get T1,T2,T3,T4 from context_weighting_vector_T
        context_weighting_vector_T = context_weighting_vector_T.unsqueeze(1)
        T1 = context_weighting_vector_T[:, :, 0]
        T2 = context_weighting_vector_T[:, :, 1]
        T3 = context_weighting_vector_T[:, :, 2]
        T4 = context_weighting_vector_T[:, :, 3]

        # obtain alpha and beta
        alpha = gmt * gvt * (T2 + T3) * (1 - T4)
        beta = (1 - gmt) * gvt * (T2 + T3) * (1 - T4)

        # get W
        W = alpha * ma + Wt_sequential * beta

        #create added object

        # create memory to be added by computing outer product
        added_memory = W[..., None] * vo[..., None, :]

        all_ones = torch.ones(batch_size, 1, self.dim).type(app_state.dtype)
        J = torch.ones(batch_size, self.slots, self.dim).type(app_state.dtype)

        # create memory to be erased by computing outer product
        erased_memory = W[..., None] * all_ones

        # Update history
        visual_working_memory = visual_working_memory * (J - erased_memory) + added_memory

        # get convolved tensor
        convolved_Wt_sequential=torch.cat((Wt_sequential[:,1:Wt_sequential.size(1)],Wt_sequential[:,0:1]),dim=1)

        # final expression to update Wt_sequential
        # print(f'{Wt_sequential.size(), convolved_Wt_sequential.size(), beta.size()}')
        Wt_sequential = (convolved_Wt_sequential * beta) + (Wt_sequential * (1 - beta))

        # final read vector
        context_read_vector = T1 * now_context + T2 * last_context + T3 * latest_context


        return context_read_vector, visual_working_memory, Wt_sequential