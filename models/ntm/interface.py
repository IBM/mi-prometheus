#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lstm_controller.py: pytorch module implementing NTM interface to external memory."""
__author__ = "Tomasz Kornuta"


import torch 
import torch.nn.functional as F
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
        self.read_param_locations = self.calculate_param_locations({'query_vector': self.num_memory_bits, 'beta': 1, 'gate': 1,  
            'shift': self.interface_shift_size, 'gamma': 1},  "Read")
        assert num_read_params == self.read_param_locations[-1], "Last location must be equal to number of read params."
        
       # Forward linear layers that generate parameters of read heads.
        self.hidden2read_list = []
        for _ in range(self.interface_num_read_heads):
            self.hidden2read_list.append(torch.nn.Linear(self.ctrl_hidden_state_size,  num_read_params))
 
        # -------------- WRITE HEAD -----------------#
        # Number/size of wrrite parameters:  key [MEMORY_BITS] + beta [1] + gate [1] + gamma [1] + 
        # + shift kernel size [SHIFT_SIZE] + erase vector [MEMORY_BITS] + write vector[MEMORY_BITS]  
        num_write_params = 3*self.num_memory_bits +1 +1 +1 +self.interface_shift_size
        
        # Write parameters - used during slicing.
        self.write_param_locations = self.calculate_param_locations({'query_vector': self.num_memory_bits, 'beta': 1, 'gate': 1,  
            'shift': self.interface_shift_size, 'gamma': 1, 
            'erase_vector': self.num_memory_bits, 'add_vector': self.num_memory_bits}, "Write")
        assert num_write_params == self.write_param_locations[-1], "Last location must be equal to number of write params."
        
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
            # Read attention weights [BATCH_SIZE x MEMORY_ADDRESSES]
            read_attentions.append(torch.ones((batch_size, num_memory_addresses), dtype=torch.float)*1e-6)
        
        # Single write head - write attention weights [BATCH_SIZE x MEMORY_ADDRESSES]
        write_attention = torch.ones((batch_size, num_memory_addresses), dtype=torch.float)*1e-6

        return InterfaceStateTuple(read_attentions,  write_attention)

    def forward(self, ctrl_hidden_state_BxH,  prev_memory_BxMxA,  prev_state_tuple):
        """
        Controller forward function. 
        
        :param ctrl_hidden_state_BxH: a Tensor with controller hidden state of size [BATCH_SIZE  x HIDDEN_SIZE]
        :param prev_memory_BxMxA: Previous state of the memory [BATCH_SIZE x MEMORY_BITS x MEMORY_ADDRESSES] 
        :param prev_state_tuple: Tuple containing previous read and write attention vectors.
        :returns: Read vector, updated memory and state tuple (object of LSTMStateTuple class).
        """
        # Unpack previous cell  state - just to make sure that everything is ok...
        (prev_read_attentions,  prev_write_attention) = prev_state_tuple
        
        # !! Execute single step !!
        
        # Read heads.
        read_attentions = []
        for i in range(self.interface_num_read_heads):
            # Calculate parameters of given read head.
            params_BxP = self.hidden2read_list[i](ctrl_hidden_state_BxH)

            # Split the parameters.
            query_vector,  beta,  gate, shift, gamma = self.split_params(params_BxP,  self.read_param_locations)

            # Update the attention of a given head.
            attention_BxA = self.update_attention(query_vector,  beta,  gate, shift, gamma,  prev_memory_BxMxA,  prev_read_attentions[i])
            # Read vector from memory.
            # TODO
            
            # Save attention.
            read_attentions.append(attention_BxA)
        
        # Write head operation.
        # TODO
          
        # TODO:  REMOVE THOSE LINES.
        write_attention = prev_write_attention
        memory_BxMxA = prev_memory_BxMxA
        
        # Pack current cell state.
        state_tuple = InterfaceStateTuple(read_attentions,  write_attention)
        
        # Return read vector, new memory state and state tuple.
        return 0, memory_BxMxA,  state_tuple
 
    def calculate_param_locations(self,  param_sizes_dict,  head_name):
        """ Calculates locations of parameters, that will subsequently be used during parameter splitting.
        :param param_sizes_dict: Dictionary containing parameters along with their sizes (in bits/units).
        :param head_name: Name of head.
        :returns: "Locations" of parameters.
        """
        logger.debug("{} param sizes dict: {}".format(head_name, param_sizes_dict))        
        # Create the parameter lengths and store their cumulative sum
        lengths = np.fromiter(param_sizes_dict.values(), dtype=int)
        # Store "parameter locations" for further usage.
        param_locations = np.cumsum(np.insert(lengths, 0, 0), dtype=int).tolist()
        logger.debug("{} param locations: {}".format(head_name, param_locations))          
        return param_locations
        
    def split_params(self,  params,  locations):
        """ Split parameters into list on the basis of locations."""
        param_splits = [params[..., locations[i]:locations[i+1]]  for i in range(len(locations)-1)]
        logger.debug("Splitted params: {}".format(param_splits)) 
        return param_splits

    def update_attention(self,  query_vector_BxM,  beta_Bx1,  gate_Bx1, shift, gamma,  prev_memory_BxMxA,  prev_attention_BxM):
        """ Updates the attention weights.
        
        :param query_vector_BxM: Query used for similarity calculation in content-based addressing [BATCH_SIZE x MEMORY_BITS]
        :param beta_Bx1: Strength parameter used in content-based addressing.
        :param prev_memory_BxMxA: a 3-D Tensor containing memory before update [BATCH_SIZE x MEMORY_BITS x MEMORY_ADDRESSES]
        """
        # Apply non-linear transformations.
        query_vector_BxM = F.tanh(query_vector_BxM) # I didn't have that one in TF!
        beta_Bx1 = F.softplus(beta_Bx1)
        gate_Bx1 = F.sigmoid(gate_Bx1)
        #gamma_Bx1 = tf.nn.softplus(params[:, self.slot_size+2:self.slot_size+3], name='gamma')
        #shift_Bx3 = tf.nn.softmax(params[:, self.slot_size+3:], name='shift')
        
        # Content-based addressing.
        attention_BxA = self.focusing_by_content(query_vector_BxM,  beta_Bx1,  prev_memory_BxMxA)
        
        # TODO: rest;)
        
        return attention_BxA
        
    def focusing_by_content(self,  query_vector_BxM, beta_Bx1, prev_memory_BxMxA):
        """Computes content-based addressing. Uses query vectors for calculation of similarity.
        
        :param query_vector_BxM: a 2-D Tensor [BATCH_SIZE x MEMORY_BITS] 
        :param beta_Bx1: a 1-D Tensor - key strength [BATCH_SIZE x 1]
        :param prev_memory_BxMxA: a 3-D Tensor containing memory before update [BATCH_SIZE x MEMORY_BITS x MEMORY_ADDRESSES]
        """
        # Normalize query batch - along samples.
        norm_query_vector_BxM = F.normalize(query_vector_BxM, p=2,  dim=1)
        logger.debug("norm_query_vector_BxM: {}".format(norm_query_vector_BxM))  
        # Normalize memory - along addresses. 
        norm_memory_BxMxA = F.normalize(prev_memory_BxMxA, p=2,  dim=1)
        logger.debug("norm_memory_BxMxA: {}".format(norm_memory_BxMxA))  
        
        # Calculate cosine similarity [BATCH_SIZE x MEMORY_ADDRESSES].
        similarity_BxA = torch.matmul(norm_query_vector_BxM.unsqueeze(1), norm_memory_BxMxA).squeeze(1)
        logger.debug("similarity_BxA: {}".format(similarity_BxA))    
        
        # Element-wise multiplication [BATCH_SIZE x MEMORY_ADDRESSES]
        strengthtened_similarity_BxA = beta_Bx1 * similarity_BxA
        logger.debug("strengthtened_similarity_BxA: {}".format(strengthtened_similarity_BxA))    

        # Calculate attention based on similarity along the "slot dimension" [BATCH_SIZE x MEMORY_ADDRESSES].
        attention_BxA = F.softmax(strengthtened_similarity_BxA, dim=1)
        logger.debug("attention_BxA: {}".format(attention_BxA))    
        return attention_BxA

