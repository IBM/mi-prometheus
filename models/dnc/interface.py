import torch
import torch.nn.functional as F
import numpy as np
import collections
from models.dnc.tensor_utils import circular_conv, normalize
from models.dnc.memory import Memory

# Helper collection type.
_InterfaceStateTuple = collections.namedtuple('InterfaceStateTuple', ('read_weights', 'write_weights'))

class InterfaceStateTuple(_InterfaceStateTuple):
    """Tuple used by interface for storing current/past state information"""
    __slots__ = ()

CUDA = False
dtype = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


class Interface:
    def __init__(self, params):
        """Initialize Interface.

        :param num_heads: number of heads
        :param is_cam [boolean]: are the heads allowed to use content addressing
        :param num_shift: number of shifts of heads.
        :param M: Number of slots per address in the memory bank.
        """
        self.num_heads = params["num_heads"]
        #self.M = M

        # Define a dictionary for attentional parameters
        self.is_cam = params["use_content_addressing"]
    
        # Parse parameters.
        # Set input and hidden  dimensions.
        self.input_size = params["control_bits"] + params["data_bits"]
        self.ctrl_hidden_state_size = params['hidden_state_dim']
        # Get memory parameters.
        self.num_memory_bits = params['memory_content_size']
        # TODO - move memory size somewhere?
        self.num_memory_addresses = params['memory_addresses_size']


    @property
    def read_size(self):
        """
        Returns the size of the data read by all heads
        
        :return: (num_head*content_size)
        """
        return self.num_heads * self.num_memory_bits

    def read(self, prev_interface_tuple, mem):
        """returns the data read from memory

        :param wt: the read weight [batch_size, memory_size]
        :param mem: the memory [batch_size, content_size, memory_size] 
        :return: the read data [batch_size, content_size]
        """
        (wt, _) = prev_interface_tuple
        #print(wt.shape)
        memory = Memory(mem)
        #print(mem.shape)
        read_data = memory.attention_read(wt)
        # flatten the data_gen in the last 2 dimensions
        sz = read_data.size()[:-2]
        #print(read_data.shape)
        #print(self.read_size)
        return read_data.view(*sz, self.read_size)

    def edit_memory(self, interface_tuple,update_data , mem):
        """returns the data read from memory

        :param wt: the read weight [batch_size, memory_size]
        :param mem: the memory [batch_size, content_size, memory_size] 
        :return: the read data [batch_size, content_size]
        """
                
        (_,  write_attention) = interface_tuple

        # Write to memory
        write_gate=update_data['write_gate']
        add=update_data['write_vectors']
        erase=update_data['erase_vectors']

         
        add=write_gate*add
 
        memory = Memory(mem)

        memory.erase_weighted(erase, write_attention)
        memory.add_weighted(add, write_attention)

        mem = memory.content

        return mem



    def init_state(self,  batch_size):
        """
        Returns 'zero' (initial) state tuple.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - object of InterfaceStateTuple class.
        """
        # Read attention weights [BATCH_SIZE x MEMORY_SIZE]
        read_attention = torch.ones((batch_size, self.num_heads, self.num_memory_addresses)).type(dtype)*1e-6
        # Write attention weights [BATCH_SIZE x MEMORY_SIZE]
        write_attention = torch.ones((batch_size, self.num_heads,  self.num_memory_addresses)).type(dtype)*1e-6

        return InterfaceStateTuple(read_attention,  write_attention)


    def update_weight(self,wt,memory,beta,g,k,s,gamma):
        if self.is_cam:
            wt_k = memory.content_similarity(k)       # content addressing ...
            wt_beta = F.softmax(beta * wt_k, dim=-1)                # ... modulated by β
            wt = g * wt_beta + (1 - g) * wt              # scalar interpolation
        wt_s = circular_conv(wt, s)                   # convolution with shift
        #wt_s = wt                   # convolution with shift

        eps = 1e-12
        wt = (wt_s + eps) ** gamma
        wt = normalize(wt)                    # sharpening with normalization

        return wt



    def update(self, update_data, prev_interface_tuple, mem):
        """Erases from memory, writes to memory, updates the weights using various attention mechanisms
        :param update_data: the parameters from the controllers [update_size]
        :param wt: the read weight [BATCH_SIZE, MEMORY_SIZE]
        :param mem: the memory [BATCH_SIZE, CONTENT_SIZE, MEMORY_SIZE] 
        :return: TUPLE [wt, mem]
        """
        #assert update_data.size()[-1] == self.update_size, "Mismatch in update sizes"

        # reshape update data_gen by heads and total parameter size
        #sz = update_data.size()[:-1]
        #update_data = update_data.view(*sz, self.num_heads, self.cum_lengths[-1])

        # split the data_gen according to the different parameters
        #data_splits = [update_data[..., self.cum_lengths[i]:self.cum_lengths[i+1]]
        #               for i in range(len(self.cum_lengths)-1)]

        (prev_read_attention,  prev_write_attention) = prev_interface_tuple

        # Obtain update parameters
        s=update_data['shifts']
        γ=update_data['sharpening']
        sr=update_data['shifts_read']
        γr=update_data['sharpening_read']
                 
        #s, γ, erase, add = data_splits
        if self.is_cam:
            k=update_data['write_content_keys']
            β=update_data['write_content_strengths']
            kr=update_data['read_content_keys']
            βr=update_data['read_content_strengths']

            g=update_data['allocation_gate']
            #temporary(this will actually be used in the usage operation in the final version)
            #The exact equivalent is the read_mode
            gr=update_data['free_gate']


        #retrieve memory Class
        memory = Memory(mem)

        #update attention       
        read_attention=self.update_weight(prev_read_attention, memory,β,g,k,s,γ)
        write_attention=self.update_weight(prev_write_attention, memory,βr,gr,kr,sr,γr)


               #if torch.sum(torch.abs(torch.sum(wt[:,0,:], dim=-1) - 1.0)) > 1e-6:
        #    print("error: gamma very high, normalization problem")
        interface_state_tuple = InterfaceStateTuple(read_attention,  write_attention)
        return interface_state_tuple




