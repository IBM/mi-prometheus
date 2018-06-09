import torch
from torch import nn
from models.dwm.controller import Controller
from models.dwm.interface import Interface
import collections

CUDA = False
dtype = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

# Helper collection type.
_DWMCellStateTuple = collections.namedtuple('DWMStateTuple', ('ctrl_state', 'interface_state',  'memory_state'))

class DWMCellStateTuple(_DWMCellStateTuple):
    """Tuple used by NTM Cells for storing current/past state information"""
    __slots__ = ()

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
        self.M = M

        # build the interface and controller
        self.interface = Interface(num_heads, is_cam, num_shift, M)
        self.controller = Controller(in_dim, output_units, state_units,
                                     self.interface.read_size, self.interface.update_size)

    def init_state(self, memory_addresses_size, batch_size, dtype):

        # Initialize controller state.
        ctrl_init_state = self.controller.init_state(batch_size, dtype)

        # Initialize interface state.
        interface_init_state = self.interface.init_state(memory_addresses_size, batch_size, dtype)

        # Initialize memory
        mem_init = (torch.ones((batch_size, self.M, memory_addresses_size)) * 0.01).type(dtype)

        return DWMCellStateTuple(ctrl_init_state, interface_init_state, mem_init)

    def forward(self, input, tuple_cell_state_prev):
        """
        Builds the DWM cell
        
        :param input: Current input (from time t)  [BATCH_SIZE x INPUT_SIZE]
        :param state: Previous hidden state (from time t-1)  [BATCH_SIZE x STATE_UNITS]
        :return: Tuple [output, hidden_state]
        """
        tuple_ctrl_state_prev, tuple_interface_prev, mem_prev = tuple_cell_state_prev

        # step1: read from memory using attention
        read_data = self.interface.read(tuple_interface_prev.head_weight, mem_prev)

        # step2: controller
        output, ctrl_state, update_data = self.controller(input, tuple_ctrl_state_prev, read_data)

        # step3: update memory and attention
        tuple_interface, mem = self.interface.update(update_data, tuple_interface_prev, mem_prev)

        tuple_cell_state = DWMCellStateTuple(ctrl_state, tuple_interface, mem)
        return output, tuple_cell_state
