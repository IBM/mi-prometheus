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

"""ntm_cell.py: pytorch module implementing single (recurrent) cell of Neural Turing Machine"""
__author__ = "Tomasz Kornuta"

import torch
import collections

from miprometheus.models.controllers.controller_factory import ControllerFactory
from miprometheus.models.ntm.ntm_interface import NTMInterface

# Helper collection type.
_NTMCellStateTuple = collections.namedtuple(
    'NTMCellStateTuple',
    ('ctrl_state',
     'interface_state',
     'memory_state',
     'read_vectors'))


class NTMCellStateTuple(_NTMCellStateTuple):
    """
    Tuple used by NTM Cells for storing current/past state information.
    """
    __slots__ = ()


class NTMCell(torch.nn.Module):
    """
    Class representing a single NTM cell.
    """

    def __init__(self, params):
        """
        Cell constructor. Cell creates controller and interface. It also
        initializes memory "block" that will be passed between states.

        :param params: Dictionary of parameters.

        """
        # Call constructor of base class.
        super(NTMCell, self).__init__()

        # Parse parameters.
        # Set input and output sizes.
        self.input_size = params['input_item_size']
        self.output_size = params['output_item_size']

        # Get controller hidden state size.
        self.controller_hidden_state_size = params['controller'][
            'hidden_state_size']

        # Get memory parameters - required by initialization of ext_controller
        # input size.
        self.num_memory_content_bits = params['memory']['num_content_bits']

        # Get number of read heads.
        self.interface_num_read_heads = params['interface']['num_read_heads']

        # Controller - entity that processes input and produces hidden state of the ntm cell.
        # controller_input_size = input_size + read_vector_size * num_read_heads
        ext_controller_inputs_size = self.input_size + \
            self.num_memory_content_bits * self.interface_num_read_heads
        # Create dictionary wirh controller parameters.
        controller_params = params['controller']
        controller_params.add_default_params({
            "input_size": ext_controller_inputs_size,
            "output_size": self.controller_hidden_state_size
        })
        # Build the controller.
        self.controller = ControllerFactory.build(controller_params)
        # Interface - entity responsible for accessing the memory.
        self.interface = NTMInterface(params)

        # Layer that produces output on the basis of... hidden state?
        ext_hidden_size = self.controller_hidden_state_size + \
            self.num_memory_content_bits * self.interface_num_read_heads
        self.hidden2output = torch.nn.Linear(ext_hidden_size, self.output_size)

    def init_state(self, init_memory_BxAxC):
        """
        Returns 'zero' (initial) state. "Recursivelly" calls controller and
        interface initialization.

        :param init_memory_BxAxC: Initial memory.
        :returns: Initial state tuple - object of NTMCellStateTuple class.

        """
        batch_size = init_memory_BxAxC.size(0)
        num_memory_addresses = init_memory_BxAxC.size(1)

        # Initialize controller state.
        ctrl_init_state = self.controller.init_state(batch_size)

        # Initialize interface state.
        interface_init_state = self.interface.init_state(
            batch_size, num_memory_addresses)

        # Initialize read vectors - one for every head.
        # Unpack cell state.
        (init_read_state_tuples, _) = interface_init_state
        (init_read_attentions_BxAx1_H, _, _, _) = zip(*init_read_state_tuples)

        read_vectors_BxC_H = []
        for h in range(self.interface_num_read_heads):
            # Read vector [BATCH_SIZE x CONTENT_BITS]
            #read_vectors_BxC_H.append(torch.zeros((batch_size, self.num_memory_content_bits)).type(dtype))
            # Read vectors from memory using the initial attention.
            read_vectors_BxC_H.append(self.interface.read_from_memory(
                init_read_attentions_BxAx1_H[h], init_memory_BxAxC))

        # Pack and return a tuple.
        ntm_state = NTMCellStateTuple(
            ctrl_init_state,
            interface_init_state,
            init_memory_BxAxC,
            read_vectors_BxC_H)
        return ntm_state

    # def init_state_from_prev_state(self, prev_cell_state):
        # """
        # Creates 'zero' (initial) state on the basis of he previous cell state.
        # "Recursivelly" calls controller and interface initialization.
        #
        #:param prev_cell_state: Previous cell state.
        #:returns: Initial state tuple - object of NTMCellStateTuple class.
        # """
        # Unpack previous cell state
        #prev_ctrl_state, prev_interface_state, prev_memory_BxAxC, prev_read_vectors_BxC_H = prev_cell_state

        # Initialize controller state.
        #ctrl_init_state =  self.controller.init_state_from_state(prev_ctrl_state)

        # Initialize interface state.
        #interface_init_state =  self.interface.init_state_from_state(prev_interface_state)

        # Pack and return a tuple.
        #ntm_state = NTMCellStateTuple(prev_ctrl_state, prev_interface_state,  prev_memory_BxAxC, prev_read_vectors_BxC_H)
        # return ntm_state

    def forward(self, inputs_BxI, prev_cell_state):
        """
        Forward function of NTM cell.

        :param inputs_BxI: a Tensor of input data of size [BATCH_SIZE  x INPUT_SIZE]
        :param  prev_cell_state: a NTMCellStateTuple tuple, containing previous state of the cell.
        :returns: an output Tensor of size  [BATCH_SIZE x OUTPUT_SIZE] and  NTMCellStateTuple tuple containing current cell state.

        """
        # Unpack previous cell  state.
        (prev_ctrl_state_tuple, prev_interface_state_tuple,
         prev_memory_BxAxC, prev_read_vectors_BxC_H) = prev_cell_state

        # Concatenate inputs with previous read vectors [BATCH_SIZE x (INPUT + NUM_HEADS * MEMORY_CONTENT_BITS)]
        #print("prev_read_vectors_BxC_H =", prev_read_vectors_BxC_H[0].size())
        prev_read_vectors = torch.cat(prev_read_vectors_BxC_H, dim=1)
        #print("inputs_BxI =", inputs_BxI.size())
        #print("prev_read_vectors =", prev_read_vectors.size())
        controller_input = torch.cat((inputs_BxI, prev_read_vectors), dim=1)

        # Execute controller forward step.
        ctrl_output_BxH, ctrl_state_tuple = self.controller(
            controller_input, prev_ctrl_state_tuple)

        # Execute interface forward step.
        read_vectors_BxC_H, memory_BxAxC, interface_state_tuple = self.interface(
            ctrl_output_BxH, prev_memory_BxAxC, prev_interface_state_tuple)

        # Output layer - takes controller output concateneted with new read
        # vectors.
        read_vectors = torch.cat(read_vectors_BxC_H, dim=1)
        ext_hidden = torch.cat((ctrl_output_BxH, read_vectors), dim=1)
        logits_BxO = self.hidden2output(ext_hidden)

        # Pack current cell state.
        cell_state_tuple = NTMCellStateTuple(
            ctrl_state_tuple,
            interface_state_tuple,
            memory_BxAxC,
            read_vectors_BxC_H)

        # Return logits and current cell state.
        return logits_BxO, cell_state_tuple
