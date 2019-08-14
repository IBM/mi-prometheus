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
memory_update_unit.py: Implementation of the ``MemoryUpdateUnit`` for the VWM network.
"""

__author__ = "Vincent Albouy, T.S. Jayram"

import torch


def memory_update(visual_object, memory_attention,
                  visual_working_memory, wt_sequential,
                  do_replace, do_add_new):
    """
    Memory update.

    :param visual_object:  [batch_size x dim]
    :param memory_attention:  [batch_size x N_v x dim]
    :param visual_working_memory: [batch_size x (H*W) x dim]
    :param wt_sequential: attention over VWM [batch_size x N_v]
    :param do_replace: replace existing object in VWM? [batch_size]
    :param do_add_new: or add new visual object     [batch_size]

    :return: new_visual_working_memory
    :return: new_wt_sequential
    """

    # pad extra dimension
    do_replace = do_replace[..., None]
    do_add_new = do_add_new[..., None]

    # get attention on the correct slot in memory based on the 2 predicates
    # attention defaults to 0 if neither condition holds
    wt = do_replace * memory_attention + do_add_new * wt_sequential

    # Update visual_working_memory
    new_visual_working_memory = (visual_working_memory * (1 - wt)[..., None]
                                 + wt[..., None] * visual_object[..., None, :])

    # compute shifted sequential head to right
    shifted_wt_sequential = torch.roll(wt_sequential, shifts=1, dims=-1)

    # new sequential attention
    new_wt_sequential = ((shifted_wt_sequential * do_add_new)
                         + (wt_sequential * (1 - do_add_new)))

    return new_visual_working_memory, new_wt_sequential
