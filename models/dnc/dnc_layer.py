import torch
from torch import nn
from torch.autograd import Variable

from models.model_base import ModelBase
from models.dnc.dnc_cell import DNCCell
from misc.app_state import AppState
import numpy as np

class DNC(ModelBase, nn.Module):

    def __init__(self, params):
        """Initialize an DNC Layer.

        :param in_dim: input size.
        :param output_units: output size.
        :param state_units: state size.
        :param num_heads: number of heads.
        :param is_cam: is it content_addressable.
        :param num_shift: number of shifts of heads.
        :param M: Number of slots per address in the memory bank.
        """
        self.in_dim = params["control_bits"] + params["data_bits"]
        try:
            self.output_units  = params['output_bits']
        except KeyError:
            self.output_units = params['data_bits']

        self.state_units =params["hidden_state_dim"]
        self.is_cam = params["use_content_addressing"]
        self.num_shift = params["shift_size"]
        self.M = params["memory_content_size"]
        #self.batch_size = params["batch_size"]
        self.memory_addresses_size = params["memory_addresses_size"]
        self.label = params["name"]
        self.app_state = AppState()
        self.cell_state_history = None 
        super(DNC, self).__init__()

        # Create the DNC components
        self.DNCCell = DNCCell(self.in_dim, self.output_units, self.state_units,
                               self.is_cam, self.num_shift, self.M,params)

    def forward(self, data_tuple):       # x : batch_size, seq_len, input_size
        """
        Runs the DNC cell and plots if necessary
        
        :param x: input sequence  [BATCH_SIZE x seq_len x input_size ]
        :param state: Input hidden state  [BATCH_SIZE x state_size]
        :return: Tuple [output, hidden_state]
        """

        (inputs, targets) = data_tuple

        dtype = torch.cuda.FloatTensor if inputs.is_cuda else torch.FloatTensor

        output = None

        if self.app_state.visualize:
            self.cell_state_history = []

        batch_size = inputs.size(0)
        seq_length = inputs.size(1)
       
        memory_addresses_size = self.memory_addresses_size
 
        if memory_addresses_size == -1:
            memory_addresses_size = seq_length  # a hack for now

        # init state
        cell_state = self.DNCCell.init_state(memory_addresses_size,batch_size,dtype)


        #cell_state = self.init_state(memory_addresses_size)
        for j in range(seq_length):
            output_cell, cell_state = self.DNCCell(inputs[..., j, :], cell_state)

            if output_cell is None:
                continue

            output_cell = output_cell[..., None, :]
            if output is None:
                output = output_cell
            else:
                output = torch.cat([output, output_cell], dim=-2)

            #if self.plot_active:
            #    self.plot_memory_attention(output, cell_state)

        return output

    def plot_memory_attention(self, output, states):
        # plot attention/memory

        from models.dnc.plot_data import plot_memory_attention
        plot_memory_attention(output, states[2], states[1][0], states[1][1], states[1][2], self.label)


