import torch
from torch import nn
import torch.nn.functional as F
from models.dwm.controller import Controller
from models.dwm.interface import Interface

from models.dwm.tensor_utils import normalize


class NTMCell(nn.Module):
    def __init__(self, dwm_in_dim, dwm_output_units, dwm_state_units,
                 num_heads, is_cam, num_shift, M):
        """Initialize an NTM cell.

        :param dwm_in_dim: input size.
        :param dwm_output_units: output size.
        :param dwm_state_units: state size.
        :param num_heads: number of heads.
        :param is_cam: is it content_addressable.
        :param num_shift: number of shifts of heads.
        :param M: Number of slots per address in the memory bank.
        """
        super(NTMCell, self).__init__()
        self.num_heads = num_heads

        # build the interface and controller
        self.interface = Interface(num_heads, is_cam, num_shift, M)
        self.controller = Controller(dwm_in_dim, dwm_output_units, dwm_state_units,
                                     self.interface.read_size, self.interface.update_size)

    def forward(self, dwm_input, state):
        """
        Builds the NTM cell
        
        :param dwm_input: Current input (from time t)  [BATCH_SIZE x INPUT_SIZE]
        :param state: Previous hidden state (from time t-1)  [BATCH_SIZE x dwm_STATE_UNITS]
        :return: Tuple [output, hidden_state]
        """
        dwm_state, wt, wt_dynamic, mem = state

        # step1: read from memory using attention
        read_data = self.interface.read(wt, mem)

        # step2: controller
        dwm_output, dwm_state, update_data = self.controller(dwm_input, dwm_state, read_data)

        # step3: update memory and attention
        wt, wt_dynamic, mem = self.interface.update(update_data, wt, wt_dynamic, mem)

        state = dwm_state, wt, wt_dynamic, mem
        return dwm_output, state
