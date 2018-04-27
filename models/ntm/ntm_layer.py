import torch
from torch import nn
from problems.plot_data import plot_memory_attention
from torch.autograd import Variable

from models.ntm.ntm_cell import NTMCell

CUDA = False
dtype = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

class NTM(nn.Module):

    def __init__(self, params):
        """Initialize an NTM Layer.

        :param tm_in_dim: input size.
        :param tm_output_units: output size.
        :param tm_state_units: state size.
        :param num_heads: number of heads.
        :param is_cam: is it content_addressable.
        :param num_shift: number of shifts of heads.
        :param M: Number of slots per address in the memory bank.
        """
        self.tm_in_dim = params["control_bits"] + params["data_bits"]
        self.tm_output_units = params["data_bits"]
        self.tm_state_units =params["hidden_state_dim"]
        self.num_heads = params["num_heads"]
        self.is_cam = params["use_content_addressing"]
        self.num_shift = params["shift_size"]
        self.M = params["memory_content_size"]
        self.batch_size = params["batch_size"]
        self.memory_addresses_size = params["memory_addresses_size"]
        self.plot_active = params["plot_memory"]
        self.label = params["name"]
        super(NTM, self).__init__()

        # Create the NTM components
        self.NTMCell = NTMCell(self.tm_in_dim, self.tm_output_units, self.tm_state_units,
                               self.num_heads, self.is_cam, self.num_shift, self.M)

    def forward(self, x):       # x : batch_size, seq_len, input_size
        """
        Runs the NTM cell and plots if necessary
        
        :param x: input sequence  [BATCH_SIZE x seq_len x input_size ]
        :param state: Input hidden state  [BATCH_SIZE x state_size]
        :return: Tuple [output, hidden_state]
        """
        output = None
        memory_addresses_size = self.memory_addresses_size
        states = self.init_state(memory_addresses_size)
        for j in range(x.size()[-2]):
            tm_output, states = self.NTMCell(x[..., j, :], states)

            if self.plot_active:
                self.plot_memory_attention(states)

            if tm_output is None:
                continue

            tm_output = tm_output[..., None, :]
            if output is None:
                output = tm_output
                continue

            # concatenate output
            output = torch.cat([output, tm_output], dim=-2)

        return output

    def init_state(self, memory_addresses_size):

        tm_state = Variable(torch.ones((self.batch_size, self.tm_state_units)).type(dtype))

        # initial attention  vector
        wt = Variable(torch.zeros((self.batch_size, self.num_heads, memory_addresses_size)).type(dtype))
        wt[:, 0:self.num_heads, 0] = 1.0

        # bookmark
        wt_dynamic = wt

        mem_t = Variable((torch.ones((self.batch_size, self.M, memory_addresses_size)) * 0.01).type(dtype))

        states = [tm_state, wt, wt_dynamic, mem_t]
        return states

    def plot_memory_attention(self, states):
        # plot attention/memory
        plot_memory_attention(states[3], states[1], self.label)


