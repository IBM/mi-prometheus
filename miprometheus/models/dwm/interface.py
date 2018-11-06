#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""interface.py: Controlls the reading and writing from memory with the various DWM attention mechanisms"""
__author__ = " Younes Bouhadjar, T.S Jayram"

import torch
import numpy as np
import logging
import collections

from miprometheus.models.dwm.tensor_utils import circular_conv, normalize
from miprometheus.models.dwm.memory import Memory
from miprometheus.utils.app_state import AppState

# Helper collection type.
_InterfaceStateTuple = collections.namedtuple(
    'InterfaceStateTuple', ('head_weight', 'snapshot_weight'))


class InterfaceStateTuple(_InterfaceStateTuple):
    """
    Tuple used by interface for storing current/past interface information:

    head_weight and snapshot_weight.

    """
    __slots__ = ()


logger = logging.getLogger('DWM_interface')


class Interface:
    """
    Implementation of the interface of the DWM.
    """

    def __init__(self, num_heads, is_cam, num_shift, M):
        """
        Initialize Interface.

        :param num_heads: number of heads
        :param is_cam (boolean): are the heads allowed to use content addressing
        :param num_shift: number of shifts of heads.
        :param M: Number of slots per address in the memory bank.

        """
        self.num_heads = num_heads
        self.M = M

        # Define a dictionary for attentional parameters
        self.is_cam = is_cam
        self.param_dict = {'s': num_shift, 'jd': 1,
                           'j': 3, 'γ': 1, 'erase': M, 'add': M}
        if self.is_cam:
            self.param_dict.update({'k': M, 'β': 1, 'g': 1})

        # create the parameter lengths and store their cumulative sum
        lengths = np.fromiter(self.param_dict.values(), dtype=int)
        self.cum_lengths = np.cumsum(
            np.insert(lengths, 0, 0), dtype=int).tolist()

    def init_state(self, memory_addresses_size, batch_size):
        """
        Returns 'zero' (initial) state of Interface tuple.

        :param batch_size: Size of the batch in given iteraction/epoch.
        :param memory_addresses_size: size of the memory

        :returns: Initial state tuple - object of InterfaceStateTuple class: (head_weight_init, snapshot_weight_init)

        """
        dtype = AppState().dtype

        # initial attention  vector
        head_weight_init = torch.zeros(
            (batch_size, self.num_heads, memory_addresses_size)).type(dtype)
        head_weight_init[:, 0:self.num_heads, 0] = 1.0

        # bookmark
        snapshot_weight_init = head_weight_init

        return InterfaceStateTuple(head_weight_init, snapshot_weight_init)

    @property
    def read_size(self):
        """
        Returns the size of the data read by all heads.

        :return: (num_head*content_size)

        """
        return self.num_heads * self.M

    @property
    def update_size(self):
        """
        Returns the total number of parameters output by the controller.

        :return: (num_heads*parameters_per_head)

        """
        return self.num_heads * self.cum_lengths[-1]

    def read(self, wt, mem):
        """
        Returns the data read from memory.

        :param wt: head's weights [batch_size, num_heads, memory_addresses_size]
        :param mem: the memory content [batch_size, memory_content_size, memory_addresses_size]

        :return: the read data [batch_size, num_heads, memory_content_size]

        """

        memory = Memory(mem)
        read_data = memory.attention_read(wt)
        # flatten the data_gen in the last 2 dimensions
        sz = read_data.size()[:-2]
        return read_data.view(*sz, self.read_size)

    def update(self, update_data, tuple_interface_prev, mem):
        """
        Erases from memory, writes to memory, updates the weights using various
        attention mechanisms.

        :param update_data: the parameters from the controllers
        :param tuple_interface_prev: contains (head_weight, snapshot_weight)
        :param tuple_interface_prev.head_weight: head attention [batch_size, num_heads, memory_size]
        :param tuple_interface_prev.snapshot_weight: snapshot(bookmark) attention [batch_size, num_heads, memory_size]
        :param mem: the memory [batch_size, content_size, memory_size]

        :returns: InterfaceTuple contains [head_weight, snapshot_weight]: the updated weight of head and snapshot
        :returns: mem: the new memory content

        """
        wt_head_prev, wt_att_snapshot_prev = tuple_interface_prev
        assert update_data.size(
        )[-1] == self.update_size, "Mismatch in update sizes"

        # reshape update data_gen by heads and total parameter size
        sz = update_data.size()[:-1]
        update_data = update_data.view(
            *sz, self.num_heads, self.cum_lengths[-1])

        # split the data_gen according to the different parameters
        data_splits = [
            update_data
            [..., self.cum_lengths[i]: self.cum_lengths[i + 1]]
            for i in range(len(self.cum_lengths) - 1)]

        # Obtain update parameters
        if self.is_cam:
            s, jd, j, γ, erase, add, k, β, g = data_splits
            # Apply Activations
            # key vector used for content-based addressing
            k = torch.nn.functional.tanh(k)
            # key strength used for content-based addressing
            β = torch.nn.functional.softplus(β)
            g = torch.nn.functional.sigmoid(g)               # interpolation gate
        else:
            s, jd, j, γ, erase, add = data_splits

        # shift weighting (determines how the weight is rotated)
        s = torch.nn.functional.softmax(torch.nn.functional.softplus(s), dim=-1)
        γ = 1 + torch.nn.functional.softplus(γ)                   # used for weight sharpening
        erase = torch.nn.functional.sigmoid(erase)                # erase memory content

        # Write to memory
        memory = Memory(mem)
        memory.erase_weighted(erase, wt_head_prev)
        memory.add_weighted(add, wt_head_prev)

        # update attention
        #  Set jumping mechanisms

        #  fixed attention to address 0
        wt_address_0 = torch.zeros_like(wt_head_prev)
        wt_address_0[:, :, 0] = 1

        # interpolation between wt and wt_d
        jd = torch.nn.functional.sigmoid(jd)
        wt_att_snapshot = (1 - jd) * wt_head_prev + jd * wt_att_snapshot_prev

        # interpolation between wt_0 wt_d wt
        j = torch.nn.functional.softmax(j, dim=-1)
        j = j[:, :, None, :]

        wt_head = j[..., 0] * wt_head_prev \
            + j[..., 1] * wt_att_snapshot \
            + j[..., 2] * wt_address_0

        # Move head according to content based addressing and shifting
        if self.is_cam:
            # content addressing ...
            wt_k = memory.content_similarity(k)
            # ... modulated by β
            wt_β = torch.nn.functional.softmax(β * wt_k, dim=-1)
            # scalar interpolation
            wt_head = g * wt_β + (1 - g) * wt_head

        # convolution with shift
        wt_s = circular_conv(wt_head, s)

        eps = 1e-12
        wt_head = (wt_s + eps) ** γ
        # sharpening with normalization
        wt_head = normalize(wt_head)

        # check attention is invalid for head 0
        check_wt = torch.max(
            torch.abs(torch.sum(wt_head[:, 0, :], dim=-1) - 1.0))
        if check_wt > 1.0e-5:
            logger.warning("Warning: gamma very high, normalization problem")

        mem = memory.content
        return InterfaceStateTuple(wt_head, wt_att_snapshot), mem
