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

"""dwm_cell.py: The cell of the DWM. It operates on a single word"""
__author__ = "Younes Bouhadjar"

import torch
from torch import nn
from models.dwm.controller import Controller
from models.dwm.interface import Interface
import collections
from misc.app_state import AppState

# Helper collection type.
_DWMCellStateTuple = collections.namedtuple(
    'DWMStateTuple', ('ctrl_state', 'interface_state', 'memory_state'))


class DWMCellStateTuple(_DWMCellStateTuple):
    """Tuple used by DWM Cells for storing current/past state information: controller state, interface state, memory state """
    __slots__ = ()


class DWMCell(nn.Module):
    """Applies the DWM cell to an element in the input sequence """

    def __init__(self, in_dim, output_units, state_units,
                 num_heads, is_cam, num_shift, M):
        """Builds the DWM cell

            :param in_dim: input size.
            :param output_units: output size.
            :param state_units: state size.
            :param num_heads: number of heads.
            :param is_cam: is it content_address    able.
            :param num_shift: number of shifts of heads.
            :param M: Number of slots per address in the memory bank.

        """

        super(DWMCell, self).__init__()
        self.num_heads = num_heads
        self.M = M

        # build the interface and controller
        self.interface = Interface(num_heads, is_cam, num_shift, M)
        self.controller = Controller(
            in_dim,
            output_units,
            state_units,
            self.interface.read_size,
            self.interface.update_size)

    def init_state(self, memory_addresses_size, batch_size):
        dtype = AppState().dtype

        # Initialize controller state.
        tuple_ctrl_init_state = self.controller.init_state(batch_size)

        # Initialize interface state.
        tuple_interface_init_state = self.interface.init_state(
            memory_addresses_size, batch_size)

        # Initialize memory
        mem_init = (torch.ones(
            (batch_size, self.M, memory_addresses_size)) * 0.01).type(dtype)

        return DWMCellStateTuple(
            tuple_ctrl_init_state, tuple_interface_init_state, mem_init)

    def forward(self, input, tuple_cell_state_prev):
        """
        forward pass of the DWM_Cell

        :param input: current input (from time t) [batch_size, inputs_size]
        :param tuple_cell_state_prev: contains (tuple_ctrl_state_prev, tuple_interface_prev, mem_prev), object of class DWMCellStateTuple

        :return: output: logits [batch_size, output_size]
        :return: tuple_cell_state: contains (tuple_ctrl_state, tuple_interface, mem)


        .. math::

            step1: read memory

            r_t &= M_t \otimes w_t

            step2: controller

            h_t &= \sigma(W_h[x_t,h_{t-1},r_{t-1}])

            y_t &= W_{y}[x_t,h_{t-1},r_{t-1}]

            P_t &= W_{P}[x_t,h_{t-1},r_{t-1}]

            step3: memory update

            M_t &= M_{t-1}\circ (E-w_t \otimes e_t)+w_t\otimes a_t

            to be completed ...

        """

        tuple_ctrl_state_prev, tuple_interface_prev, mem_prev = tuple_cell_state_prev

        # step1: read from memory using attention
        read_data = self.interface.read(
            tuple_interface_prev.head_weight, mem_prev)

        # step2: controller
        output, tuple_ctrl_state, update_data = self.controller(
            input, tuple_ctrl_state_prev, read_data)

        # step3: update memory and attention
        tuple_interface, mem = self.interface.update(
            update_data, tuple_interface_prev, mem_prev)

        tuple_cell_state = DWMCellStateTuple(
            tuple_ctrl_state, tuple_interface, mem)
        return output, tuple_cell_state
