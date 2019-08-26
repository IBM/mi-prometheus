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


def memory_update(visual_working_memory, write_head,
                  visual_object, read_head, do_replace, do_add_new):
    """
    Memory update.

    :param visual_working_memory: [batch_size x (H*W) x dim]
    :param write_head: attention over VWM [batch_size x N_v]
    :param visual_object:  [batch_size x dim]
    :param read_head:  [batch_size x N_v x dim]
    :param do_replace: replace existing object in VWM? [batch_size]
    :param do_add_new: or add new visual object     [batch_size]

    :return: new_visual_working_memory
    :return: new_write_head
    """

    # pad extra dimension
    do_replace = do_replace[..., None]
    do_add_new = do_add_new[..., None]

    # get attention on the correct slot in memory based on the 2 predicates
    # attention defaults to 0 if neither condition holds
    wt = do_replace * read_head + do_add_new * write_head

    # Update visual_working_memory
    new_visual_working_memory = (visual_working_memory * (1 - wt)[..., None]
                                 + wt[..., None] * visual_object[..., None, :])

    # compute shifted sequential head to right
    shifted_write_head = torch.roll(write_head, shifts=1, dims=-1)

    # new sequential attention
    new_write_head = ((shifted_write_head * do_add_new)
                      + (write_head * (1 - do_add_new)))

    return new_visual_working_memory, new_write_head
