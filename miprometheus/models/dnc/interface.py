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

"""interface.py: Controlls the reading and writing from memory with the various DNC attention mechanisms"""
__author__ = " Ryan L. McAvoy"


import torch
import torch.nn.functional as F
import collections
from miprometheus.models.dnc.tensor_utils import circular_conv, normalize
from miprometheus.models.dnc.memory import Memory
from miprometheus.models.dnc.memory_usage import MemoryUsage
from miprometheus.models.dnc.temporal_linkage import TemporalLinkage
from miprometheus.utils.app_state import AppState

# Helper collection type.
_InterfaceStateTuple = collections.namedtuple(
    'InterfaceStateTuple', ('read_weights', 'write_weights', 'usage', 'links'))


class InterfaceStateTuple(_InterfaceStateTuple):
    """
    Tuple used by interface for storing current/past state information.
    """
    __slots__ = ()


class Interface:
    def __init__(self, params):
        """
        Initialize Interface.

        :param params: dictionary of input parameters

        """

        # Get memory parameters.
        self.num_memory_bits = params['memory_content_size']

        # Number of read and write heads
        self._num_reads = params["num_reads"]
        self._num_writes = params["num_writes"]

        # parameters that determine whether this acts as a DNC or NTM
        self.use_ntm_write = params['use_ntm_write']
        self.use_ntm_read = params['use_ntm_read']
        self.use_ntm_order = params['use_ntm_order']
        self.use_extra_write_gate = params['use_extra_write_gate']

        self.mem_usage = MemoryUsage()

        self.temporal_linkage = TemporalLinkage(self._num_writes)

    @property
    def read_size(self):
        """
        Returns the size of the data read by all heads.

        :return: (num_head*content_size)

        """
        return self._num_reads * self.num_memory_bits

    def read(self, prev_interface_tuple, mem):
        """
        returns the data read from memory.

        :param prev_interface_tuple: Tuple [previous read, previous write, prev usage, prev links]
        :param mem: the memory [batch_size, content_size, memory_size]
        :return: the read data [batch_size, content_size]

        """
        (wt, _, _, _) = prev_interface_tuple

        memory = Memory(mem)
        read_data = memory.attention_read(wt)
        # flatten the data_gen in the last 2 dimensions
        sz = read_data.size()[:-2]
        return read_data.view(*sz, self.read_size)

    def edit_memory(self, interface_tuple, update_data, mem):
        """
        Edits the external memory and then returns it.

        :param update_data: the parameters from the controllers [dictionary]
        :param prev_interface_tuple: Tuple [previous read, previous write, prev usage, prev links]
        :param mem: the memory [batch_size, content_size, memory_size]
        :return: edited memory [batch_size, content_size, memory_size]

        """

        (_, write_attention, _, _) = interface_tuple

        # Write to memory
        write_gate = update_data['write_gate']
        add = update_data['write_vectors']
        erase = update_data['erase_vectors']

        if self.use_extra_write_gate:
            add = add * write_gate
            erase = erase * write_gate

        memory = Memory(mem)

        memory.erase_weighted(erase, write_attention)
        memory.add_weighted(add, write_attention)

        mem = memory.content

        return mem

    def init_state(self, memory_address_size, batch_size):
        """
        Returns 'zero' (initial) state tuple.

        :param memory_address_size: The number of memory addresses
        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - object of InterfaceStateTuple class.

        """
        dtype = AppState().dtype

        # Read attention weights [BATCH_SIZE x MEMORY_SIZE]
        read_attention = torch.ones(
            (batch_size, self._num_reads, memory_address_size)).type(dtype) * 1e-6

        # Write attention weights [BATCH_SIZE x MEMORY_SIZE]
        write_attention = torch.ones(
            (batch_size, self._num_writes, memory_address_size)).type(dtype) * 1e-6

        # Usage of memory cells [BATCH_SIZE x MEMORY_SIZE]
        usage = self.mem_usage.init_state(memory_address_size, batch_size)

        # temporal links tuple
        link_tuple = self.temporal_linkage.init_state(
            memory_address_size, batch_size)

        return InterfaceStateTuple(
            read_attention, write_attention, usage, link_tuple)

    def update_weight(self, prev_attention, memory,
                      strength, gate, key, shift, sharp):
        """
        Update the attention with NTM's mix of content addressing and linear
        shifting.

        :param prev_attention: tensor of shape `[batch_size, num_writes,
              memory_size]` giving the attention at the previous time step.
        :param memory: the memory of the previous step (class)
        :param strength: The strengthening parameter for the content addressing [batch, num_heads, 1]
        :param gate: The interpolation gate between the content addressing and the previous weight [batch, num_heads, 1]
        :param key: The comparison key for the content addressing [batch, num_heads, num_memory_bits]
        :param shift: The shift vector that defines the circular convolution of the outputs [batch, num_heads, num_shifts]
        :param sharp: sharpening parameter for the attention [batch, num_heads, 1]

        """
        # Content addressing using weighted cosine similarity
        similarity = memory.content_similarity(key)
        content_weights = F.softmax(strength * similarity, dim=-1)

        # Gate between the current weight and the content weights
        attention = gate * content_weights + (1 - gate) * prev_attention

        # Linear shift with convolution
        shifted_attention = circular_conv(attention, shift)

        # Sharpen weights and then normalize
        eps = 1e-12
        attention = (shifted_attention + eps) ** sharp
        attention = normalize(attention)

        return attention

    def update_write_weight(self, usage, memory,
                            allocation_gate, write_gate, key, strength):
        """
        Update write attention with DNC's combination of content addressing and
        usage based allocation.

        :param usage: A tensor of shape `[batch_size, memory_size]` representing
              current memory usage.
        :param memory: the memory of the previous step (class)
        :param strength: The strengthening parameter for the content addressing [batch, num_writes, 1]
        :param key: The comparison key for the content addressing [batch, num_writes, num_memory_bits]
        :param allocation_gate:  Interpolation between writing to unallocated memory and
                                 content-based lookup, for each write head [batch, num_writes, 1]
        :param write_gate: Overall gating of write amount for each write head. [batch, num_writes, 1]

        """
        # Calculate which memory slots are open for allocation
        write_allocation_weights = self.mem_usage.write_allocation_weights(
            usage=usage,
            write_gates=(allocation_gate * write_gate),
            num_writes=self._num_writes)

        # Content addressing using weighted cosine similarity
        similarity = memory.content_similarity(key)
        content_weights = F.softmax(strength * similarity, dim=-1)

        # Gate between the allocatable memory and the content weighted memory
        wt = write_gate * (allocation_gate * write_allocation_weights +
                           (1 - allocation_gate) * content_weights)

        return wt

    def update_read_weight(
            self, link, memory, prev_read_weights, read_mode, key, strength):
        """
        Update the read attention with the DNC's combination of content
        addressing and temporal link propagation to go forwards or backwards in
        time.

        :param link: A tensor of shape `[batch_size, num_writes, memory_size,
              memory_size]` representing the previous link graphs for each write
              head.
        :param memory: the memory of the previous step (class)
        :param prev_read_weights: tensor of shape `[batch_size, num_reads,
              memory_size]` containing the previous read weights w_{t-1}^r.
        :param read_mode: Mixing between "backwards" and "forwards" positions (for each write head)
                          and content-based lookup, for each read head [batch, num_reads, 1+2*numwrites]
        :param strength: The strengthening parameter for the content addressing [batch, num_reads, 1]
        :param key: The comparison key for the content addressing [batch, num_reads, num_memory_bits]

        """
        # Content addressing using weighted cosine similarity
        similarity = memory.content_similarity(key)
        content_weights = F.softmax(strength * similarity, dim=-1)

        # Calculate the weight to go forward and backwards along the links
        # matrix
        forward_weights = self.temporal_linkage.directional_read_weights(
            link.link, prev_read_weights, forward=True)
        backward_weights = self.temporal_linkage.directional_read_weights(
            link.link, prev_read_weights, forward=False)

        # Reshape the read mode matrix
        backward_mode = torch.unsqueeze(read_mode[:, :, :self._num_writes], 3)
        forward_mode = torch.unsqueeze(
            read_mode[:, :, self._num_writes:2 * self._num_writes], 3)
        content_mode = torch.unsqueeze(
            read_mode[:, :, 2 * self._num_writes], 2)

        # Gate between the content similarity, going forwards along the link
        # matrix and going backwards
        read_weights = (content_mode * content_weights +
                        torch.sum(forward_mode * forward_weights, 2) +
                        torch.sum(backward_mode * backward_weights, 2))

        return read_weights

    def update_read(self, update_data, prev_interface_tuple, mem):
        """
        Updates the read attention switching between the NTM and DNC
        mechanisms.

        :param update_data: the parameters from the controllers [dictionary]
        :param prev_interface_tuple: Tuple [previous read, previous write, prev usage, prev links[
        :param prev_memory_BxMxA: the memory of the previous step (class)
        :return: The new interface tuple with an updated usage and write attention

        """
        (prev_read_attention, prev_write_attention,
         prev_usage, prev_links) = prev_interface_tuple

        # Parameters for the content addressing
        key = update_data['read_content_keys']
        strength = update_data['read_content_strengths']

        # retrieve memory Class
        memory = Memory(mem)

        # update the attention using either the NTM read mechanism (True) or
        # the DNC (False)
        if self.use_ntm_read:
            # Parameters for shift addressing
            shift = update_data['shifts_read']
            sharp = update_data['sharpening_read']
            gate = update_data['read_mode_shift']

            read_attention = self.update_weight(
                prev_read_attention, memory, strength, gate, key, shift, sharp)
            links = prev_links
        else:

            read_mode = update_data['read_mode']
            links = self.temporal_linkage.calc_temporal_links(
                prev_write_attention, prev_links)
            read_attention = self.update_read_weight(
                links, memory, prev_read_attention, read_mode, key, strength)

        interface_state_tuple = InterfaceStateTuple(
            read_attention, prev_write_attention, prev_usage, links)
        return interface_state_tuple

    def update_write(self, update_data, prev_interface_tuple, mem):
        """
        Updates the write attention switching between the NTM and DNC
        mechanisms.

        :param update_data: the parameters from the controllers [dictionary]
        :param prev_interface_tuple: Tuple [previous read, previous write, prev usage, prev links]
        :param prev_memory_BxMxA: the memory of the previous step (class)
        :return: The new interface tuple with an updated usage and write attention

        """

        (prev_read_attention, prev_write_attention,
         prev_usage, prev_links) = prev_interface_tuple

        # Obtain update parameters

        key = update_data['write_content_keys']
        strength = update_data['write_content_strengths']
        gate = update_data['allocation_gate']

        # retrieve memory Class
        memory = Memory(mem)

        free_gate = update_data['free_gate']
        usage = self.mem_usage.calculate_usage(
            prev_write_attention, free_gate, prev_read_attention, prev_usage)

        # update the attention using either the NTM write mechanism (True) or
        # the DNC (False)
        if self.use_ntm_write:
            # Parameters for shift addressing
            shift = update_data['shifts']
            sharp = update_data['sharpening']

            write_attention = self.update_weight(
                prev_write_attention, memory, strength, gate, key, shift, sharp)
        else:
            write_gate = update_data['write_gate']
            allocation_gate = gate
            write_attention = self.update_write_weight(
                usage, memory, allocation_gate, write_gate, key, strength)

        interface_state_tuple = InterfaceStateTuple(
            prev_read_attention, write_attention, usage, prev_links)
        return interface_state_tuple

    def update_and_edit(self, update_data,
                        prev_interface_tuple, prev_memory_BxMxA):
        """
        Erases from memory, writes to memory, updates the weights using various
        attention mechanisms.

        :param update_data: the parameters from the controllers [update_size]
        :param prev_interface_tuple: the read weight [BATCH_SIZE, MEMORY_SIZE]
        :param prev_memory_BxMxA: the memory of the previous step (class)
        :return: the new read vector, the update memory, the new interface tuple

        """

        (prev_read_attention, prev_write_attention,
         prev_usage, prev_links) = prev_interface_tuple

        # Step 1: update the write weights
        interface_tuple = self.update_write(
            update_data, prev_interface_tuple, prev_memory_BxMxA)

        # Step 2: Write and Erase Data
        memory_BxMxA = self.edit_memory(
            interface_tuple, update_data, prev_memory_BxMxA)

        # Step 3: Update read weights using either the current or previous
        # memory
        if self.use_ntm_order:
            read_memory_BxMxA = prev_memory_BxMxA
        else:
            read_memory_BxMxA = memory_BxMxA

        interface_tuple = self.update_read(
            update_data, interface_tuple, read_memory_BxMxA)

        # Step 4: Read the data from memory
        read_vector_BxM = self.read(interface_tuple, memory_BxMxA)

        return read_vector_BxM, memory_BxMxA, interface_tuple
