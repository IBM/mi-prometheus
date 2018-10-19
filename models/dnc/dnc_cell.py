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

"""dnc_cell.py: The cell of the DNC. It operates on a single word"""
__author__ = "Ryan L. McAvoy, Tomasz Kornuta"


import torch
from torch import nn
import collections
from models.dnc.control_and_params import ControlParams
from models.dnc.interface import Interface
from utils.app_state import AppState

# Helper collection type.
_NTMCellStateTuple = collections.namedtuple(
    'NTMStateTuple',
    ('ctrl_init_state',
     'int_init_state',
     'memory_state',
     'read_vector'))


class NTMCellStateTuple(_NTMCellStateTuple):
    """
    Tuple used by NTM Cells for storing current/past state information.
    """
    __slots__ = ()


class DNCCell(nn.Module):
    """
    Class representing a single cell of the DNC.
    """
    def __init__(self, output_size, params):
        """
        Initialize an DNC cell.

        :param output_size: output size.
        :param state_units: state size.
        :param num_heads: number of heads.

        """
        super(DNCCell, self).__init__()

        # Get memory parameters.
        self.num_memory_bits = params['memory_content_size']

        # build the interface and controller
        self.interface = Interface(params)
        self.control_params = ControlParams(
            output_size, self.interface.read_size, params)
        self.output_network = nn.Linear(self.interface.read_size, output_size)

    def init_state(self, memory_address_size, batch_size):
        """
        Returns 'zero' (initial) state:

        * memory  is reset to random values.
        * read & write weights (and read vector) are set to 1e-6.

        :param batch_size: Size of the batch in given iteraction/epoch.
        :param num_memory_adresses: Number of memory addresses.

        """
        dtype = AppState().dtype

        # Initialize controller state.
        ctrl_init_state = self.control_params.init_state(batch_size)

        # Initialize interface state.
        interface_init_state = self.interface.init_state(
            memory_address_size, batch_size)

        # Memory [BATCH_SIZE x MEMORY_BITS x MEMORY_SIZE]
        init_memory_BxMxA = torch.zeros(
            batch_size, self.num_memory_bits, memory_address_size).type(dtype)

        # Read vector [BATCH_SIZE x MEMORY_SIZE]
        read_vector_BxM = self.interface.read(
            interface_init_state, init_memory_BxMxA)

        # Pack and return a tuple.
        return NTMCellStateTuple(
            ctrl_init_state,
            interface_init_state,
            init_memory_BxMxA,
            read_vector_BxM)

    def forward(self, input_BxI, cell_state_prev):
        """
        Builds the DNC cell.

        :param input: Current input (from time t)  [BATCH_SIZE x INPUT_SIZE]
        :param state: Previous hidden state (from time t-1)  [BATCH_SIZE x STATE_UNITS]
        :return: Tuple [output, hidden_state]

        """
        ctrl_state_prev_tuple, prev_interface_tuple, prev_memory_BxMxA, prev_read_vector_BxM = cell_state_prev

        # Step 1: controller
        output_from_hidden, ctrl_state_tuple, update_data = self.control_params(
            input_BxI, ctrl_state_prev_tuple, prev_read_vector_BxM)

        read_vector_BxM, memory_BxMxA, interface_tuple = self.interface.update_and_edit(
            update_data, prev_interface_tuple, prev_memory_BxMxA)

        # generate final output data from controller output and the read data
        final_output = output_from_hidden + \
            self.output_network(read_vector_BxM)

        cell_state = NTMCellStateTuple(
            ctrl_state_tuple, interface_tuple, memory_BxMxA, read_vector_BxM)
        return final_output, cell_state
