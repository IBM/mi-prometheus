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

    def forward(self, valid_vo, valid_mo, vo, mo, ma, vwm,
                temporal_class_weights, wt_sequential):
        """
        Forward pass of the ``MemoryUpdateUnit``. 
        
        :param valid_vo : whether visual object is valid
        :param valid_mo : whether memory object is valid
        :param vo : visual object
        :param mo: memory object
        :param ma: memory attention
        :param vwm: visual working memory
        :param temporal_class_weights: matrix to get T1,T2,T3,T4
        :param wt_sequential :wt_sequential
        
        :return: context_read_vector, vwm, wt_sequential
        """

        batch_size = vo.size(0)

        # get T1,T2,T3,T4 from temporal_class_weights
        # corresponds to now, last, latest, or none
        T1 = temporal_class_weights[:, 0]
        T2 = temporal_class_weights[:, 1]
        T3 = temporal_class_weights[:, 2]
        T4 = temporal_class_weights[:, 3]

        # obtain alpha and beta
        alpha = valid_mo * valid_vo * (T2 + T3) * (1 - T4)
        beta = (1 - valid_mo) * valid_vo * (T2 + T3) * (1 - T4)

        # pad extra dimension
        alpha = alpha[..., None]
        beta = beta[..., None]

        # get w
        w = alpha * ma + beta * wt_sequential

        # create memory to be added by computing outer product
        added_memory = w[..., None] * vo[..., None, :]

        all_ones = torch.ones(batch_size, 1, self.dim).type(app_state.dtype)
        J = torch.ones(batch_size, self.slots, self.dim).type(app_state.dtype)

        # create memory to be erased by computing outer product
        erased_memory = w[..., None] * all_ones

        # Update history
        visual_working_memory = vwm * (J - erased_memory) + added_memory

        # compute shifted sequential head to right
        shifted_wt_sequential = torch.roll(wt_sequential, shifts=1, dims=-1)

        # new sequential attention
        new_wt_sequential = (shifted_wt_sequential * beta) \
                            + (wt_sequential * (1 - beta))

        # final read vector
        # (now or latest) and valid visual object?
        is_visual = (T1 + T3) * valid_vo
        # optional extra check that it is neither last nor none
        # is_visual = is_visual * (1 - T2) * (1 - T4)

        # (now or (latest and (not valid visual object))) and valid memory object?
        is_mem = (T2 + T3 * (1 - valid_vo)) * valid_mo
        # optional extra check that it is neither now nor none
        # is_mem = is_mem * (1 - T1) * (1 - T4)

        # output correct object for reasoning
        output_object = is_visual[..., None] * vo + is_mem[..., None] * mo

        #print(output_object[0], visual_working_memory[0], new_wt_sequential[0])

        return output_object, visual_working_memory, new_wt_sequential
