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

"""temporal_linkage.py: Class governing the main read mechanism of the DNC.
                    This is a Pytorch conversion of the TemporalLinkage class from Deepmind's Tensorflow implementation
                    of the DNC (Copyright 2017 Google Inc.) https://github.com/deepmind/dnc/blob/master/addressing.py
"""

__author__ = " Ryan L. McAvoy"

import torch
import collections
from mip.utils.app_state import AppState

_TemporalLinkageState = collections.namedtuple('TemporalLinkageState',
                                               ('link', 'precedence_weights'))


class TemporalLinkageState(_TemporalLinkageState):
    """
    Tuple used by interface for storing current/past state information.
    """
    __slots__ = ()


class TemporalLinkage():
    """
    Keeps track of write order for forward and backward addressing. This is a
    pseudo-RNNCore module, whose state is a pair `(link, precedence_weights)`,
    where `link` is a (collection of) graphs for (possibly multiple) write
    heads (represented by a tensor with values in the range.

    [0, 1]), and `precedence_weights` records the "previous write
    locations" used to build the link graphs. The function
    `directional_read_weights` computes addresses following the forward
    and backward directions in the link graphs.

    """

    def __init__(self, num_writes, name='temporal_linkage'):
        """
        Construct a TemporalLinkage module. Args:

        :param memory_size: The number of memory slots.
        :param num_writes: The number of write heads.
        :param name: Name of the module.

        """
        super(TemporalLinkage, self).__init__()
        self._num_writes = num_writes

    def init_state(self, memory_address_size, batch_size):
        """
        Returns 'zero' (initial) state tuple.

        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - object of InterfaceStateTuple class.

        """
        dtype = AppState().dtype
        self._memory_size = memory_address_size
        link = torch.ones(
            (batch_size,
             self._num_writes,
             memory_address_size,
             memory_address_size)).type(dtype) * 1e-6

        precendence_weights = torch.ones(
            (batch_size, self._num_writes, memory_address_size)).type(dtype) * 1e-6

        return TemporalLinkageState(link, precendence_weights)

    def calc_temporal_links(self, write_weights, prev_state):
        """
        Calculate the updated linkage state given the write weights.

        Args:
          :param write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
              containing the memory addresses of the different write heads.
          :param prev_state: `TemporalLinkageState` tuple containg a tensor `link` of
              shape `[batch_size, num_writes, memory_size, memory_size]`, and a
              tensor `precedence_weights` of shape `[batch_size, num_writes,
              memory_size]` containing the aggregated history of recent writes.

        Returns:
          :returns: A `TemporalLinkageState` tuple `next_state`, which contains the updated
          link and precedence weights.

        """
        link = self._link(prev_state.link, prev_state.precedence_weights,
                          write_weights)
        precedence_weights = self._precedence_weights(
            prev_state.precedence_weights, write_weights)
        return TemporalLinkageState(
            link=link, precedence_weights=precedence_weights)

    def directional_read_weights(self, link, prev_read_weights, forward):
        """
        Calculates the forward or the backward read weights.

        For each read head (at a given address), there are `num_writes` link graphs
        to follow. Thus this function computes a read address for each of the
        `num_reads * num_writes` pairs of read and write heads.

        Args:
          :param link: tensor of shape `[batch_size, num_writes, memory_size, memory_size]` representing the link graphs L_t.

          :param prev_read_weights: tensor of shape `[batch_size, num_reads, memory_size]` containing the previous read weights w_{t-1}^r.

          :param forward: Boolean indicating whether to follow the "future" direction in the link graph (True) or the "past" direction (False).

        Returns:
          :returns: tensor of shape `[batch_size, num_reads, num_writes, memory_size]`

        """
        # We calculate the forward and backward directions for each pair of
        # read and write heads; hence we need to tile the read weights and do a
        # sort of "outer product" to get this.
        expanded_read_weights = torch.stack(
            [prev_read_weights] * self._num_writes, dim=1)
        if forward:
            link = torch.transpose(link, 2, 3)
        result = torch.matmul(expanded_read_weights, link)

        # reverse the transpose
        # Swap dimensions 1, 2 so order is [batch, reads, writes, memory]:
        result_t = torch.transpose(result, 1, 2)
        return result_t

    def _link(self, prev_link, prev_precedence_weights, write_weights):
        """
        Calculates the new link graphs. For each write head, the link is a
        directed graph (represented by a matrix with entries in range [0, 1])
        whose vertices are the memory locations, and an edge indicates temporal
        ordering of writes. Args:

          :param prev_link: A tensor of shape `[batch_size, num_writes, memory_size,
              memory_size]` representing the previous link graphs for each write
              head.
          :param prev_precedence_weights: A tensor of shape `[batch_size, num_writes,
              memory_size]` which is the previous "aggregated" write weights for
              each write head.
          :param write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
              containing the new locations in memory written to.
        Returns:
          :returns: A tensor of shape `[batch_size, num_writes, memory_size, memory_size]`
          containing the new link graphs for each write head.

        """
        batch_size = prev_link.shape[0]
        write_weights_i = torch.unsqueeze(write_weights, 3)
        write_weights_j = torch.unsqueeze(write_weights, 2)

        prev_precedence_weights_j = torch.unsqueeze(prev_precedence_weights, 2)

        prev_link_scale = 1 - write_weights_i - write_weights_j
        new_link = write_weights_i * prev_precedence_weights_j
        link = prev_link_scale * prev_link + new_link
        # Return the link with the diagonal set to zero, to remove self-looping
        # edges.
        # this is the messiest way to handle this. Need a better way to set equal to zero
        # unfortunately the diag function in pytorch does not handle batches
        for i in range(batch_size):
            for j in range(self._num_writes):
                diagonal = torch.diag(link[i, j, :, :])
                link[i, j, :, :] = link[i, j, :, :] - torch.diag(diagonal)

        return link

    def _precedence_weights(self, prev_precedence_weights, write_weights):
        """
        Calculates the new precedence weights given the current write weights.
        The precedence weights are the "aggregated write weights" for each
        write head, where write weights with sum close to zero will leave the
        precedence weights unchanged, but with sum close to one will replace
        the precedence weights. Args:

          :param prev_precedence_weights: A tensor of shape `[batch_size, num_writes,
              memory_size]` containing the previous precedence weights.
          :param write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
              containing the new write weights.
        Returns:
          :returns: A tensor of shape `[batch_size, num_writes, memory_size]` containing the
          new precedence weights.

        """
        write_sum = torch.sum(write_weights, 2, keepdim=True)
        precedence_weights = (1 - write_sum) * \
            prev_precedence_weights + write_weights
        return precedence_weights

 #   @property
 #   def state_size(self):
 #       """Returns a `TemporalLinkageState` tuple of the state tensors' shapes."""
 #       return TemporalLinkageState(
 #           link=tf.TensorShape(
 #               [self._num_writes, self._memory_size, self._memory_size]),
 #           precedence_weights=tf.TensorShape([self._num_writes,
 #                                              self._memory_size]),)
