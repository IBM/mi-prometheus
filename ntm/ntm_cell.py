import torch
from torch import nn
import torch.nn.functional as F

from ntm.controller import Controller
from ntm.interface import Interface


class NTMCell(nn.Module):
    def __init__(self, tm_in_dim, tm_output_units, tm_state_units,
                 num_heads, is_cam, num_shift, M):
        """Initialize an NTM cell.

        :param tm_in_dim: input size.
        :param tm_output_units: output size.
        :param tm_state_units: state size.
        :param num_heads: number of heads.
        :param is_cam: is it content_addressable.
        :param num_shift: number of shifts of heads.
        :param M: Number of slots per address in the memory bank.
        """
        super(NTMCell, self).__init__()
        tm_state_in = tm_state_units + tm_in_dim
        self.tm_i2w = nn.Linear(tm_state_in, num_heads)

        # build the interface and controller
        self.interface = Interface(num_heads, is_cam, num_shift, M)
        self.controller = Controller(tm_in_dim, tm_output_units, tm_state_units,
                                     self.interface.read_size, self.interface.update_size)

    def forward(self, tm_input, state):
        tm_state, wt, mem = state

        # step 0 : shift to address 0?
        combined = torch.cat((tm_state, tm_input), dim=-1)
        f = self.tm_i2w(combined)
        f = F.sigmoid(f)
        wt_address_0 = torch.zeros_like(wt)
        wt_address_0[:, :, 0] = 1
        f = f[...,None]
        wt = f * wt_address_0 + (1 - f) * wt

        # step1: read from memory using attention
        read_data = self.interface.read(wt, mem)

        # step2: controller
        tm_output, tm_state, update_data = self.controller(tm_input, tm_state, read_data)

        # step3: update memory and attention
        wt, mem = self.interface.update(update_data, wt, mem)

        state = tm_state, wt, mem
        return tm_output, state
