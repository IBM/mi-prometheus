import torch
from torch import nn
from models.dnc.plot_data import plot_memory_attention
from torch.autograd import Variable

from models.model_base import ModelBase
from models.dnc.dnc_cell import DNCCell

CUDA = False
dtype = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

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
        self.output_units = params["data_bits"]
        self.state_units =params["hidden_state_dim"]
        self.num_heads = params["num_heads"]
        self.is_cam = params["use_content_addressing"]
        self.num_shift = params["shift_size"]
        self.M = params["memory_content_size"]
        #self.batch_size = params["batch_size"]
        self.memory_addresses_size = params["memory_addresses_size"]
        self.label = params["name"]
        super(DNC, self).__init__()

        # Create the DNC components
        self.DNCCell = DNCCell(self.in_dim, self.output_units, self.state_units,
                               self.num_heads, self.is_cam, self.num_shift, self.M,params)

    def forward(self, inputs):       # x : batch_size, seq_len, input_size
        """
        Runs the DNC cell and plots if necessary
        
        :param x: input sequence  [BATCH_SIZE x seq_len x input_size ]
        :param state: Input hidden state  [BATCH_SIZE x state_size]
        :return: Tuple [output, hidden_state]
        """
        output = None

        batch_size = inputs.size(0)
        seq_length = inputs.size(1)
       
        memory_addresses_size = self.memory_addresses_size
 
        if memory_addresses_size == -1:
            memory_addresses_size = seq_length  # a hack for now

        # init state
        cell_state = self.DNCCell.init_state(batch_size)


        #cell_state = self.init_state(memory_addresses_size)
        for j in range(seq_length):
            output_cell, cell_state = self.DNCCell(inputs[..., j, :], cell_state)

            if output_cell is None:
                continue

            output_cell = output_cell[..., None, :]
            if output is None:
                output = output_cell
                continue

            # concatenate output
            output = torch.cat([output, output_cell], dim=-2)

            #if self.plot_active:
            #    self.plot_memory_attention(output, cell_state)

        return output

    def init_state(self, memory_addresses_size,batch_size):

        state = Variable(torch.ones((batch_size, self.state_units)).type(dtype))

        # initial attention  vector
        wt = Variable(torch.zeros((batch_size, self.num_heads, memory_addresses_size)).type(dtype))
        wt[:, 0:self.num_heads, 0] = 1.0

        # bookmark
        wt_dynamic = wt

        mem_t = Variable((torch.ones((batch_size, self.M, memory_addresses_size)) * 0.01).type(dtype))

        states = [state, wt, wt_dynamic, mem_t]
        return states

    def plot_memory_attention(self, output, states):
        # plot attention/memory
        plot_memory_attention(output, states[3], states[1], states[2], self.label)


