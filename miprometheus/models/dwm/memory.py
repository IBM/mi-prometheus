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

"""memory.py: Class for editing memory """
__author__ = "Younes Bouhadjar"

import torch
from miprometheus.models.dwm.tensor_utils import sim, outer_prod


class Memory:
    """
    Implementation of the memory of the DWM.
    """
    def __init__(self, mem_t):
        """
        Initializes the memory.

        :param mem_t: the memory at time t [batch_size, memory_content_size, memory_addresses_size]

        """

        self._memory = mem_t

    def attention_read(self, wt):
        """
        Returns the data read from memory.

        :param wt: head's weights [batch_size, num_heads, memory_addresses_size]
        :return: the read data [batch_size, num_heads, memory_content_size]

        """

        return sim(wt, self._memory)

    def add_weighted(self, add, wt):
        """
        Writes data to memory.

        :param wt: head's weights [batch_size, num_heads, memory_addresses_size]
        :param add: the data to be added to memory [batch_size, num_heads, memory_content_size]

        :return the updated memory [batch_size, memory_addresses_size, memory_content_size]

        """

        # memory = memory + sum_{head h} weighted add(h)
        self._memory = self._memory + torch.sum(outer_prod(add, wt), dim=-3)

    def erase_weighted(self, erase, wt):
        """
        Erases elements from memory.

        :param wt: head's weights [batch_size, num_heads, memory_addresses_size]
        :param erase: data to be erased from memory [batch_size, num_heads, memory_content_size]

        :return the updated memory [batch_size, memory_addresses_size, memory_content_size]

        """

        # memory = memory * product_{head h} (1 - weighted erase(h))
        self._memory = self._memory * \
            torch.prod(1 - outer_prod(erase, wt), dim=-3)

    def content_similarity(self, k):
        """
        Calculates the dot product for Content aware addressing.

        :param k: the keys emitted by the controller [batch_size, num_heads, memory_content_size]
        :return: the dot product between the keys and query [batch_size, num_heads, memory_addresses_size]

        """

        return sim(k, self._memory, l2_normalize=True, aligned=False)

    @property
    def size(self):
        """
        Returns the size of the memory.

        :return: Int size of the memory

        """

        return self._memory.size()

    @property
    def content(self):
        """
        Returns the entire memory.

        :return: the memory []

        """

        return self._memory
