import torch
from torch import nn
import collections
from models.dnc.controller import Controller
from models.dnc.interface import Interface

CUDA = False
dtype = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

# Helper collection type.
_NTMCellStateTuple = collections.namedtuple('NTMStateTuple', ('ctrl_init_state', 'int_init_state',  'memory_state', 'read_vector'))

class NTMCellStateTuple(_NTMCellStateTuple):
    """Tuple used by NTM Cells for storing current/past state information"""
    __slots__ = ()


class DNCCell(nn.Module):
    def __init__(self, in_dim, output_units, state_units,
                 num_heads, is_cam, num_shift, M, params):
        """Initialize an DNC cell.

        :param in_dim: input size.
        :param output_units: output size.
        :param state_units: state size.
        :param num_heads: number of heads.
        :param is_cam: is it content_addressable.
        :param num_shift: number of shifts of heads.
        :param M: Number of slots per address in the memory bank.
        """
        super(DNCCell, self).__init__()
       # self.num_heads = num_heads
        self.num_heads = params["num_heads"]
        #self.M = M

        # Define a dictionary for attentional parameters
        #self.is_cam = is_cam

        # Parse parameters.
        # Set input and hidden  dimensions.
        self.input_size = params["control_bits"] + params["data_bits"]
        self.ctrl_hidden_state_size = params['hidden_state_dim']
        # Get memory parameters.
        self.num_memory_bits = params['memory_content_size']
        # TODO - move memory size somewhere?
        self.num_memory_addresses = params['memory_addresses_size']

        #self.controller = ControllerFactory.build_model(params)


     
        # build the interface and controller
        self.interface = Interface(params)
        self.controller = Controller(in_dim, output_units, state_units,
                                     self.interface.read_size, self.num_heads, params)
        #self.controller = Controller(in_dim, output_units, state_units,
        #                             self.interface.read_size, self.num_heads)
        self.output_network = nn.Linear(self.interface.read_size, output_units)

    def init_state(self,  batch_size):
        """
        Returns 'zero' (initial) state:
        * memory  is reset to random values.
        * read & write weights (and read vector) are set to 1e-6.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :param num_memory_adresses: Number of memory addresses.
        """
        # Initialize controller state.
        ctrl_init_state =  self.controller.init_state(batch_size)

        # Initialize interface state. 
        interface_init_state =  self.interface.init_state(batch_size)

        # Memory [BATCH_SIZE x MEMORY_BITS x MEMORY_SIZE] 
        init_memory_BxMxA = torch.empty(batch_size,  self.num_memory_bits,  self.num_memory_addresses)
        torch.nn.init.normal_(init_memory_BxMxA, mean=0.5, std=0.2)
        # Read vector [BATCH_SIZE x MEMORY_SIZE]
        #read_vector_BxM = torch.ones((batch_size, self.num_memory_bits)).type(dtype)*1e-6
        read_vector_BxM = self.interface.read(interface_init_state, init_memory_BxMxA)        


        # Pack and return a tuple.
        return NTMCellStateTuple(ctrl_init_state, interface_init_state,  init_memory_BxMxA, read_vector_BxM)

    def forward(self, input_BxI, cell_state_prev):
        """
        Builds the DNC cell
        
        :param input: Current input (from time t)  [BATCH_SIZE x INPUT_SIZE]
        :param state: Previous hidden state (from time t-1)  [BATCH_SIZE x STATE_UNITS]
        :return: Tuple [output, hidden_state]
        """
        ctrl_state_prev_tuple,prev_interface_tuple, prev_memory_BxMxA, prev_read_vector_BxM = cell_state_prev

        # Step 1: controller
        output_from_hidden, ctrl_state_tuple, update_data = self.controller(input_BxI, ctrl_state_prev_tuple, prev_read_vector_BxM)

        # Step 2: update memory and attention
        interface_tuple = self.interface.update(update_data, prev_interface_tuple, prev_memory_BxMxA)
       
        # Step 3: Read the data from memory
        # Edit and then read is the order used by DeepMind's public DNC implementation but I had problems getting that to converge
        read_vector_BxM = self.interface.read(interface_tuple, prev_memory_BxMxA)

        # Step 4: Write and Erase Data
        memory_BxMxA = self.interface.edit_memory(interface_tuple, update_data, prev_memory_BxMxA)
        
        #generate final output data from controller output and the read data
        final_output = output_from_hidden + self.output_network(read_vector_BxM)

        cell_state =NTMCellStateTuple(ctrl_state_tuple, interface_tuple, memory_BxMxA, read_vector_BxM)
        return final_output, cell_state