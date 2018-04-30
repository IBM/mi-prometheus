import torch
from torch import nn
import torch.nn.functional as F
from models.ntm.controller import Controller
from models.ntm.interface import Interface

from models.ntm.tensor_utils import normalize


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
        self.num_heads = num_heads

        # build the interface and controller
        self.interface = Interface(num_heads, is_cam, num_shift, M)
        self.controller = Controller(tm_in_dim, tm_output_units, tm_state_units,
                                     self.interface.read_size, self.interface.update_size)

    def forward(self, tm_input, state):
        """
        Builds the NTM cell
        
        :param tm_input: Current input (from time t)  [BATCH_SIZE x INPUT_SIZE]
        :param state: Previous hidden state (from time t-1)  [BATCH_SIZE x TM_STATE_UNITS]
        :return: Tuple [output, hidden_state]
        """
        tm_state, wt, wt_dynamic, mem = state

        # step1: read from memory using attention
        read_data = self.interface.read(wt, mem)

        # step2: controller
        tm_output, tm_state, update_data = self.controller(tm_input, tm_state, read_data)

        # step3: update memory and attention
        wt, wt_dynamic, mem = self.interface.update(update_data, wt, wt_dynamic, mem)

        state = tm_state, wt, wt_dynamic, mem
        return tm_output, state
