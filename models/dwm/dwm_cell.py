from torch import nn
from models.dwm.controller import Controller
from models.dwm.interface import Interface


class DWMCell(nn.Module):
    def __init__(self, in_dim, output_units, state_units,
                 num_heads, is_cam, num_shift, M):
        """Initialize an DWM cell.

        :param in_dim: input size.
        :param output_units: output size.
        :param state_units: state size.
        :param num_heads: number of heads.
        :param is_cam: is it content_addressable.
        :param num_shift: number of shifts of heads.
        :param M: Number of slots per address in the memory bank.
        """
        super(DWMCell, self).__init__()
        self.num_heads = num_heads

        # build the interface and controller
        self.interface = Interface(num_heads, is_cam, num_shift, M)
        self.controller = Controller(in_dim, output_units, state_units,
                                     self.interface.read_size, self.interface.update_size)

    def forward(self, input, state):
        """
        Builds the DWM cell
        
        :param input: Current input (from time t)  [BATCH_SIZE x INPUT_SIZE]
        :param state: Previous hidden state (from time t-1)  [BATCH_SIZE x STATE_UNITS]
        :return: Tuple [output, hidden_state]
        """
        state, wt, wt_dynamic, mem = state

        # step1: read from memory using attention
        read_data = self.interface.read(wt, mem)

        # step2: controller
        output, state, update_data = self.controller(input, state, read_data)

        # step3: update memory and attention
        wt, wt_dynamic, mem = self.interface.update(update_data, wt, wt_dynamic, mem)

        state = state, wt, wt_dynamic, mem
        return output, state
