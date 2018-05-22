import torch
import torch.nn.functional as F
import numpy as np
import collections
from models.dnc.tensor_utils import circular_conv, normalize
from models.dnc.memory import Memory
from models.dnc.memory_usage import MemoryUsage
from models.dnc.temporal_linkage import TemporalLinkage

# Helper collection type.
_InterfaceStateTuple = collections.namedtuple('InterfaceStateTuple', ('read_weights', 'write_weights', 'usage', 'links'))

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
        self._num_writes = params["num_writes"]
        self._num_reads = params["num_reads"]

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
        self.use_ntm_write = params['use_ntm_write']
        self.use_ntm_order = params['use_ntm_order']
        self.use_ntm_read = params['use_ntm_read']
        self.use_extra_write_gate = params['use_extra_write_gate']

        self.mem_usage=MemoryUsage(self.num_memory_addresses)

        self.temporal_linkage=TemporalLinkage(self.num_memory_addresses,self._num_writes)

    @property
    def read_size(self):
        """
        Returns the size of the data read by all heads
        
        :return: (num_head*content_size)
        """
        return self._num_reads * self.num_memory_bits

    def read(self, prev_interface_tuple, mem):
        """returns the data read from memory

        :param wt: the read weight [batch_size, memory_size]
        :param mem: the memory [batch_size, content_size, memory_size] 
        :return: the read data [batch_size, content_size]
        """
        (wt, _,_,_) = prev_interface_tuple
        
        memory = Memory(mem)
        read_data = memory.attention_read(wt)
        #return 
        # flatten the data_gen in the last 2 dimensions
        sz = read_data.size()[:-2]
        return read_data.view(*sz, self.read_size)

    def edit_memory(self, interface_tuple,update_data , mem):
        """returns the data read from memory

        :param wt: the read weight [batch_size, memory_size]
        :param mem: the memory [batch_size, content_size, memory_size] 
        :return: the read data [batch_size, content_size]
        """
                
        (_,  write_attention,_,_) = interface_tuple

        # Write to memory
        write_gate=update_data['write_gate']
        add=update_data['write_vectors']
        erase=update_data['erase_vectors']

        if self.use_extra_write_gate:         
            add=add*write_gate
            erase=erase*write_gate
 
        memory = Memory(mem)

        memory.erase_weighted(erase, write_attention)
        memory.add_weighted(add, write_attention)

        mem = memory.content

        return mem



    def init_state(self, memory_address_size,  batch_size, dtype):
        """
        Returns 'zero' (initial) state tuple.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - object of InterfaceStateTuple class.
        """
        # Read attention weights [BATCH_SIZE x MEMORY_SIZE]
        read_attention = torch.ones((batch_size, self._num_reads, memory_address_size)).type(dtype)*1e-6
        
        # Write attention weights [BATCH_SIZE x MEMORY_SIZE]
        write_attention = torch.ones((batch_size, self._num_writes,  memory_address_size)).type(dtype)*1e-6

        # Usage of memory cells [BATCH_SIZE x MEMORY_SIZE]
        usage = self.mem_usage.init_state(memory_address_size,  batch_size,dtype)
        
        # temporal links tuple
        link_tuple = self.temporal_linkage.init_state(memory_address_size,  batch_size,dtype)

        return InterfaceStateTuple(read_attention,  write_attention,usage,link_tuple)


    def update_weight(self,wt,memory,beta,g,k,s,gamma):
        if self.is_cam:
            wt_k = memory.content_similarity(k)       # content addressing ...
            wt_beta = F.softmax(beta * wt_k, dim=-1)                # ... modulated by β
            wt = g * wt_beta + (1 - g) * wt              # scalar interpolation
        wt_s = circular_conv(wt, s)                   # convolution with shift

        eps = 1e-12
        wt = (wt_s + eps) ** gamma
        wt = normalize(wt)                    # sharpening with normalization

        return wt

    def update_write_weight(self, usage, memory, allocation_gate, write_gate, k, beta):
        # a_t^i - The allocation weights for each write head.
        write_allocation_weights = self.mem_usage.write_allocation_weights(
          usage=usage,
          write_gates=(allocation_gate * write_gate),
          num_writes=1) #remove hardcode once I prove this works

        if self.is_cam:
            wt_k = memory.content_similarity(k)       # content addressing ...
            wt_beta = F.softmax(beta * wt_k, dim=-1)                # ... modulated by β
   
        wt=write_gate * (allocation_gate * write_allocation_weights +
                           (1 - allocation_gate) * wt_beta)

        return wt 
    
    def update_read_weight(self, link, memory, prev_read_weights, read_mode, k, beta):
        if self.is_cam:
            wt_k = memory.content_similarity(k)       # content addressing ...
            content_weights = F.softmax(beta * wt_k, dim=-1)                # ... modulated by β
   
        forward_weights = self.temporal_linkage.directional_read_weights(
           link.link, prev_read_weights, forward=True)
        backward_weights = self.temporal_linkage.directional_read_weights(
          link.link, prev_read_weights, forward=False)

        
        #the unsqueezes may be unavoidable
        backward_mode = torch.unsqueeze(read_mode[:, :, :self._num_writes],3)
        forward_mode = torch.unsqueeze(read_mode[:, :, self._num_writes:2 * self._num_writes],3)
        content_mode =  torch.unsqueeze(read_mode[:, :, 2 * self._num_writes],2)

        read_weights = (content_mode * content_weights 
             + torch.sum(forward_mode * forward_weights,2) 
             + torch.sum(backward_mode * backward_weights, 2))


        return read_weights

    def update(self, update_data, prev_interface_tuple, mem):
        """Erases from memory, writes to memory, updates the weights using various attention mechanisms
        :param update_data: the parameters from the controllers [update_size]
        :param wt: the read weight [BATCH_SIZE, MEMORY_SIZE]
        :param mem: the memory [BATCH_SIZE, CONTENT_SIZE, MEMORY_SIZE] 
        :return: TUPLE [wt, mem]
        """
          

        (prev_read_attention,  prev_write_attention, prev_usage, prev_links) = prev_interface_tuple



        # Obtain update parameters
        s=update_data['shifts']
        γ=update_data['sharpening']
        sr=update_data['shifts_read']
        γr=update_data['sharpening_read']
                
        #rename variables 
        #s, γ, erase, add = data_splits
        if self.is_cam:
            k=update_data['write_content_keys']
            beta=update_data['write_content_strengths']
            kr=update_data['read_content_keys']
            betar=update_data['read_content_strengths']

            g=update_data['allocation_gate']
            #temporary(this will actually be used in the usage operation in the final version)
            #The exact equivalent is the read_mode
            #gr=update_data['free_gate']
            gr=update_data['read_mode_shift']


        #retrieve memory Class
        memory = Memory(mem)

        free_gate=update_data['free_gate']
        usage=self.mem_usage.calculate_usage(prev_write_attention, free_gate, prev_read_attention, prev_usage)
       # usage=prev_usage      


        #update attention       
        write_gate=update_data['write_gate']
        write_attention=self.update_weight(prev_write_attention, memory,beta,g,k,s,γ)
        allocation_gate=g
        #write_attention=self.update_write_weight(usage, memory, allocation_gate, write_gate, k, beta)

        #DNC actually writes before it calculates the reads

        #linkage_state = self._linkage(write_weights, prev_state.linkage)
 

        read_mode=update_data['read_mode']

        links=self.temporal_linkage.calc_temporal_links(write_attention, prev_links)
      
        read_attention=self.update_read_weight(links, memory, prev_read_attention, read_mode, kr, betar) 
        #read_attention=self.update_weight(prev_read_attention, memory,betar,gr,kr,sr,γr)
        #read_attention=self.temporal_linkage.direction(prev_read_attention, memory,betar,gr,kr,sr,γr)
    
        interface_state_tuple = InterfaceStateTuple(read_attention,  write_attention, usage,links)
        return interface_state_tuple

    def update_read(self, update_data, prev_interface_tuple, mem):
        """Erases from memory, writes to memory, updates the weights using various attention mechanisms
        :param update_data: the parameters from the controllers [update_size]
        :param wt: the read weight [BATCH_SIZE, MEMORY_SIZE]
        :param mem: the memory [BATCH_SIZE, CONTENT_SIZE, MEMORY_SIZE] 
        :return: TUPLE [wt, mem]
        """
          

        (prev_read_attention,  prev_write_attention, prev_usage, prev_links) = prev_interface_tuple



        # Obtain update parameters
        s=update_data['shifts']
        γ=update_data['sharpening']
        sr=update_data['shifts_read']
        γr=update_data['sharpening_read']
                
        #rename variables 
        #s, γ, erase, add = data_splits
        if self.is_cam:
            k=update_data['write_content_keys']
            beta=update_data['write_content_strengths']
            kr=update_data['read_content_keys']
            betar=update_data['read_content_strengths']

            g=update_data['allocation_gate']
            #temporary(this will actually be used in the usage operation in the final version)
            #The exact equivalent is the read_mode
            #gr=update_data['free_gate']
            gr=update_data['read_mode_shift']


        #retrieve memory Class
        memory = Memory(mem)

        free_gate=update_data['free_gate']
        
        read_mode=update_data['read_mode']

        if self.use_ntm_read:
            read_attention=self.update_weight(prev_read_attention, memory,betar,gr,kr,sr,γ)
            links=prev_links
        else:
            links=self.temporal_linkage.calc_temporal_links(prev_write_attention, prev_links)
            read_attention=self.update_read_weight(links, memory, prev_read_attention, read_mode, kr, betar)
        #
        #read_attention=self.temporal_linkage.direction(prev_read_attention, memory,betar,gr,kr,sr,γr)
    
        interface_state_tuple = InterfaceStateTuple(read_attention,  prev_write_attention, prev_usage,links)
        return interface_state_tuple


    def update_write(self, update_data, prev_interface_tuple, mem):
        """Erases from memory, writes to memory, updates the weights using various attention mechanisms
        :param update_data: the parameters from the controllers [update_size]
        :param wt: the read weight [BATCH_SIZE, MEMORY_SIZE]
        :param mem: the memory [BATCH_SIZE, CONTENT_SIZE, MEMORY_SIZE] 
        :return: TUPLE [wt, mem]
        """
          

        (prev_read_attention,  prev_write_attention, prev_usage, prev_links) = prev_interface_tuple



        # Obtain update parameters
        s=update_data['shifts']
        γ=update_data['sharpening']
        sr=update_data['shifts_read']
        γr=update_data['sharpening_read']
                
        #rename variables 
        #s, γ, erase, add = data_splits
        if self.is_cam:
            k=update_data['write_content_keys']
            beta=update_data['write_content_strengths']
            kr=update_data['read_content_keys']
            betar=update_data['read_content_strengths']

            g=update_data['allocation_gate']
            #temporary(this will actually be used in the usage operation in the final version)
            #The exact equivalent is the read_mode
            #gr=update_data['free_gate']
            gr=update_data['read_mode_shift']


        #retrieve memory Class
        memory = Memory(mem)

        free_gate=update_data['free_gate']
        usage=self.mem_usage.calculate_usage(prev_write_attention, free_gate, prev_read_attention, prev_usage)
       # usage=prev_usage      


        #update attention       
        write_gate=update_data['write_gate']
        if self.use_ntm_write:
            write_attention=self.update_weight(prev_write_attention, memory,beta,g,k,s,γ)
        else:
            allocation_gate=g
            write_attention=self.update_write_weight(usage, memory, allocation_gate, write_gate, k, beta)

        #DNC actually writes before it calculates the reads

        #linkage_state = self._linkage(write_weights, prev_state.linkage)
 

        interface_state_tuple = InterfaceStateTuple(prev_read_attention,  write_attention, usage,prev_links)
        return interface_state_tuple
    
    def update_and_edit(self, update_data, prev_interface_tuple, prev_memory_BxMxA):
        """Erases from memory, writes to memory, updates the weights using various attention mechanisms
        :param update_data: the parameters from the controllers [update_size]
        :param wt: the read weight [BATCH_SIZE, MEMORY_SIZE]
        :param mem: the memory [BATCH_SIZE, CONTENT_SIZE, MEMORY_SIZE] 
        :return: TUPLE [wt, mem]
        """
          

        (prev_read_attention,  prev_write_attention, prev_usage, prev_links) = prev_interface_tuple


        # Write to memory
        write_gate=update_data['write_gate']
        add=update_data['write_vectors']
        erase=update_data['erase_vectors']


        # Obtain update parameters
        s=update_data['shifts']
        γ=update_data['sharpening']
        sr=update_data['shifts_read']
        γr=update_data['sharpening_read']
                
        #rename variables 
        #s, γ, erase, add = data_splits
        if self.is_cam:
            k=update_data['write_content_keys']
            beta=update_data['write_content_strengths']
            kr=update_data['read_content_keys']
            betar=update_data['read_content_strengths']

            g=update_data['allocation_gate']
            #temporary(this will actually be used in the usage operation in the final version)
            #The exact equivalent is the read_mode
            #gr=update_data['free_gate']
            gr=update_data['read_mode_shift']


        #retrieve memory Class
        #memory = Memory(mem)

        free_gate=update_data['free_gate']
        #usage=self.mem_usage.calculate_usage(prev_write_attention, free_gate, prev_read_attention, prev_usage)
       # usage=prev_usage      


        #update attention       
        write_gate=update_data['write_gate']
        #write_attention=self.update_weight(prev_write_attention, memory,beta,g,k,s,γ)
        #allocation_gate=g
        #write_attention=self.update_write_weight(usage, memory, allocation_gate, write_gate, k, beta)

        #DNC actually writes before it calculates the reads

        #linkage_state = self._linkage(write_weights, prev_state.linkage)

        #add=add*write_gate
        #erase=erase*write_gate
 
        #memory.erase_weighted(erase, write_attention)
        #memory.add_weighted(add, write_attention)

         
        # num_bits

        read_mode=update_data['read_mode']

        #links=self.temporal_linkage.calc_temporal_links(write_attention, prev_links)
      
        #read_attention=self.update_read_weight(links, memory, prev_read_attention, read_mode, kr, betar) 
       # read_attention=self.update_weight(prev_read_attention, memory,betar,gr,kr,sr,γr)
        #read_attention=self.temporal_linkage.direction(prev_read_attention, memory,betar,gr,kr,sr,γr)
   

        interface_tuple = self.update_write(update_data, prev_interface_tuple, prev_memory_BxMxA)
        
       # interface_tuple = self.update_read(update_data, interface_tuple, prev_memory_BxMxA)
        #interface_tuple = self.update(update_data, prev_interface_tuple, prev_memory_BxMxA)     

        #interface_state_tuple = InterfaceStateTuple(read_attention,  write_attention, usage,links)
 

                # Step 3: Write and Erase Data
        memory_BxMxA = self.edit_memory(interface_tuple, update_data, prev_memory_BxMxA)


        if self.use_ntm_order:
           read_memory_BxMxA=prev_memory_BxMxA
        else: 
           read_memory_BxMxA=memory_BxMxA

        interface_tuple = self.update_read(update_data, interface_tuple, read_memory_BxMxA)

        # Step 2: update memory and attention
        #interface_tuple = self.interface.update_read(update_data, interface_tuple, memory_BxMxA)
        #interface_tuple = self.interface.update_write(update_data, interface_tuple, prev_memory_BxMxA)

        # Step 4: Read the data from memory
        read_vector_BxM = self.read(interface_tuple, memory_BxMxA)
 

 
        return read_vector_BxM, memory_BxMxA,  interface_tuple
 
