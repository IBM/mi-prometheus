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

    def forward(self, valid_vo, valid_mo, visual_object, memory_object,
                memory_attention, visual_working_memory,
                temporal_class_weights, wt_sequential,
                do_replace, do_add_new, is_visual, is_mem):
        """
        Forward pass of the ``MemoryUpdateUnit``. 
        
        :param valid_vo : whether visual object is valid
        :param valid_mo : whether memory object is valid
        :param visual_object : visual object
        :param memory_object: memory object
        :param memory_attention: memory attention
        :param visual_working_memory: visual working memory
        :param temporal_class_weights: matrix to get t1,t2,t3,t4
        :param wt_sequential :wt_sequential
        :param do_replace
        :param is_visual
        :param is_mem

        :return: relevant_object, visual_working_memory, wt_sequential
        """

        batch_size = visual_object.size(0)

        # from miprometheus.models.VWM_model.utils_VWM import eval_predicate
        # do_replace, do_add_new, is_visual, is_mem = eval_predicate(
        #     temporal_class_weights, valid_vo, valid_mo)

        # pad extra dimension
        do_replace = do_replace[..., None]
        do_add_new = do_add_new[..., None]

        # get attention on the correct slot in memory based on the 2 predicates
        # attention defaults to 0 if neither condition holds
        wt = do_replace * memory_attention + do_add_new * wt_sequential

        # Update visual_working_memory
        # visual_working_memory = (
        #         visual_working_memory
        #         + wt[..., None] * (visual_object[..., None, :] - visual_working_memory)
        #         )

        # # create memory to be added by computing outer product
        added_memory = wt[..., None] * visual_object[..., None, :]

        all_ones = torch.ones(batch_size, 1, self.dim).type(app_state.dtype)
        J = torch.ones(batch_size, self.slots, self.dim).type(app_state.dtype)

        # create memory to be erased by computing outer product
        erased_memory = wt[..., None] * all_ones

        # Update history
        visual_working_memory = visual_working_memory * (J - erased_memory) + added_memory

        # compute shifted sequential head to right
        shifted_wt_sequential = torch.roll(wt_sequential, shifts=1, dims=-1)

        # new sequential attention
        new_wt_sequential = (shifted_wt_sequential * do_add_new) \
                            + (wt_sequential * (1 - do_add_new))

        return visual_working_memory, new_wt_sequential, is_visual, is_mem

