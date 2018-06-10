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

        r"""Builds the DWM cell

        .. math::

            \begin{array}{ll}
            r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
            z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
            n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
            h' = (1 - z) * n + z * h
            \end{array}

        where :math:`\sigma` is the sigmoid function.

        Args:
            in_dim: input size.
            output_units: output size.
            state_units: state size.
            num_heads: number of heads.
            is_cam: is it content_address    able.
            num_shift: number of shifts of heads.
            M: Number of slots per address in the memory bank.

        Inputs: input, hidden
            - **input** of shape (batch_size x inputs_size): Current input (from time t)
            - tuple_cell_state_prev (hidden_state)
            hidden_state of shape (batch_size, state_units): Previous hidden state (from time t-1)

        Outputs:
            - **output** of shape `(batch_size, output_size)`:
            - **tuple_cell_state** = (ctrl_state, tuple_interface, mem)
                tuple_ctrl_state_prev: (hidden_state of shape (batch_size, state_units))
                tuple_interface: (head_weights of shape (batch_size, num_heads, memory_addresses), snapshot_weights of shape (batch_size, num_heads, memory_addresses))
                mem of shape (batch_size, memory_size_conetent)


        Examples::

        >>> dwm = DWM(params)
        >>> inputs = torch.randn(5, 3, 10)
        >>> targets = torch.randn(5, 3, 20)
        >>> data_tuple = (inputs, targets)
        >>> output = dwm(data_tuple)

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
        tuple_ctrl_init_state = self.controller.init_state(batch_size, dtype)

        # Initialize interface state.
        tuple_interface_init_state = self.interface.init_state(memory_addresses_size, batch_size, dtype)

        # Initialize memory
        mem_init = (torch.ones((batch_size, self.M, memory_addresses_size)) * 0.01).type(dtype)

        return DWMCellStateTuple(tuple_ctrl_init_state, tuple_interface_init_state, mem_init)

    def forward(self, input, tuple_cell_state_prev):

        tuple_ctrl_state_prev, tuple_interface_prev, mem_prev = tuple_cell_state_prev

        # step1: read from memory using attention
        read_data = self.interface.read(tuple_interface_prev.head_weight, mem_prev)

        # step2: controller
        output, tuple_ctrl_state, update_data = self.controller(input, tuple_ctrl_state_prev, read_data)

        # step3: update memory and attention
        tuple_interface, mem = self.interface.update(update_data, tuple_interface_prev, mem_prev)

        tuple_cell_state = DWMCellStateTuple(tuple_ctrl_state, tuple_interface, mem)
        return output, tuple_cell_state