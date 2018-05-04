#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lstm_controller.py: pytorch module implementing NTM interface to external memory."""
__author__ = "Tomasz Kornuta"


import torch 
import collections
import numpy as np
import logging
logger = logging.getLogger('NTM-Interface')

# Helper collection type.
_InterfaceStateTuple = collections.namedtuple('InterfaceStateTuple', ('read_weights', 'write_weights'))

class InterfaceStateTuple(_InterfaceStateTuple):
    """Tuple used by interface for storing current/past state information"""
    __slots__ = ()


class Interface(torch.nn.Module):
    """Class realizing interface between controller and memory.
    """
    def __init__(self, params):
        """ Constructor.
        
        :param params: Dictionary of parameters.
        """
        # Call constructor of base class.
        super(Interface, self).__init__() 

        # Parse parameters.
        # Get hidden state size.
        self.ctrl_hidden_state_size = params['ctrl_hidden_state_size']
        # Get memory parameters.
        self.num_memory_bits = params['num_memory_bits']
        # Get interface parameters.
        self.interface_num_read_heads = params['interface_num_read_heads']
        assert self.interface_num_read_heads >= 1, "NTM requires at least 1 read head (currently %r)" % self.interface_num_read_heads     

        self.interface_shift_size = params['interface_shift_size']
 
        # -------------- READ HEADS -----------------#
        # Number/size of parameters of a single read head: key [MEMORY_BITS] + beta [1] + gate [1] + gamma [1] + shift kernel size [SHIFT_SIZE]
        # All read params = NUM_HEADS * above (but it's not important here)
        num_read_params =  (self.num_memory_bits +1 +1 +1 +self.interface_shift_size)
        # Dictionary with read parameters - used during slicing.
        read_params_dict = {'key_vector': self.num_memory_bits, 'beta': 1, 'gate': 1,  'shift': self.interface_shift_size, 'gamma': 1}
        logger.debug("Read params dict: {}".format(read_params_dict))        
        # Create the parameter lengths and store their cumulative sum
        read_lengths = np.fromiter(read_params_dict.values(), dtype=int)
        # Store "parameter locations" for further usage.
        self.read_params_locations = np.cumsum(np.insert(read_lengths, 0, 0), dtype=int).tolist()
        logger.debug("Read params locations: {}".format(self.read_params_locations))    
        assert num_read_params == self.read_params_locations[-1], "Last locations must be equal to number of read params."
        
       # Forward linear layers that generate parameters of read heads.
        self.hidden2read_params_list = []
        for _ in range(self.interface_num_read_heads):
            self.hidden2read_params_list.append(torch.nn.Linear(self.ctrl_hidden_state_size,  num_read_params))
 
        # -------------- WRITE HEAD -----------------#
        # Number/size of wrrite parameters:  key [MEMORY_BITS] + beta [1] + gate [1] + gamma [1] + 
        # + shift kernel size [SHIFT_SIZE] + erase vector [MEMORY_BITS] + write vector[MEMORY_BITS]  
        num_write_params = 3*self.num_memory_bits +1 +1 +1 +self.interface_shift_size
        
        # Dictionary with write parameters - used during slicing.
        self.write_params_dict = {'key_vector': self.num_memory_bits, 'beta': 1, 'gate': 1,  'shift': self.interface_shift_size, 'gamma': 1, 
            'erase_vector': self.num_memory_bits, 'add_vector': self.num_memory_bits}
        
        
       # Forward linear layer that generates parameters of write heads.
        self.hidden2write_params = torch.nn.Linear(self.ctrl_hidden_state_size,  num_write_params)

    def init_state(self,  batch_size,  num_memory_addresses):
        """
        Returns 'zero' (initial) state tuple.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :param num_memory_addresses: Number of memory addresses.
        :returns: Initial state tuple - object of InterfaceStateTuple class.
        """
        # Add read attention vectors - one for each read head.
        read_attentions = []
        for _ in range(self.interface_num_read_heads):
            # Read attention weights [BATCH_SIZE x MEMORY_SIZE]
            read_attentions.append(torch.ones((batch_size, num_memory_addresses), dtype=torch.float)*1e-6)
        
        # Single write head - write attention weights [BATCH_SIZE x MEMORY_SIZE]
        write_attention = torch.ones((batch_size, num_memory_addresses), dtype=torch.float)*1e-6

        return InterfaceStateTuple(read_attentions,  write_attention)

    def forward(self, ctrl_hidden_state_BxH,  prev_memory_BxMxS,  prev_state_tuple):
        """
        Controller forward function. 
        
        :param ctrl_hidden_state_BxH: a Tensor with controller hidden state of size [BATCH_SIZE  x HIDDEN_SIZE]
        :param prev_memory_BxMxS: Previous state of the memory
        :param prev_state_tuple: Tuple containing previous read and write attention vectors.
        :returns: Read vector, updated memory and state tuple (object of LSTMStateTuple class).
        """
        # Unpack previous cell  state - just to make sure that everything is ok...
        (prev_read_attentions,  prev_write_attention) = prev_state_tuple
        
        # !! Execute single step !!
        # Compute parameters.
        #params_BxP = self.hidden2params(ctrl_hidden_state_BxH)

                # split the data_gen according to the different parameters
        #data_splits = [update_data[..., self.cum_lengths[i]:self.cum_lengths[i+1]] for i in range(len(self.cum_lengths)-1)]

        # TODO:  REMOVE THOSE LINES.
        read_attentions = prev_read_attentions
        write_attention = prev_write_attention
        memory_BxMxS = prev_memory_BxMxS
        
        # Pack current cell state.
        state_tuple = InterfaceStateTuple(read_attentions,  write_attention)
        
        # Return read vector, new memory state and state tuple.
        return 0, memory_BxMxS,  state_tuple
 
