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
        self.num_memory_content_bits = params['num_memory_content_bits']
        # Get interface parameters.
        self.interface_num_read_heads = params['interface_num_read_heads']
        assert self.interface_num_read_heads >= 1, "NTM requires at least 1 read head (currently %r)" % self.interface_num_read_heads     

        self.interface_shift_size = params['interface_shift_size']
 
        # -------------- READ HEADS -----------------#
        # Number/size of parameters of a single read head: key [MEMORY_CONTENT_BITS] + beta [1] + gate [1] + gamma [1] + shift kernel size [SHIFT_SIZE]
        # All read params = NUM_HEADS * above (but it's not important here)
        num_read_params =  (self.num_memory_content_bits +1 +1 +1 +self.interface_shift_size)
        # Dictionary with read parameters - used during slicing.
        self.read_param_locations = self.calculate_param_locations({'query_vector': self.num_memory_content_bits, 'beta': 1, 'gate': 1,  
            'shift': self.interface_shift_size, 'gamma': 1},  "Read")
        assert num_read_params == self.read_param_locations[-1], "Last location must be equal to number of read params."
        
       # Forward linear layers that generate parameters of read heads.
        self.hidden2read_list = []
        for _ in range(self.interface_num_read_heads):
            self.hidden2read_list.append(torch.nn.Linear(self.ctrl_hidden_state_size,  num_read_params))
 
        # -------------- WRITE HEAD -----------------#
        # Number/size of wrrite parameters:  key [MEMORY_BITS] + beta [1] + gate [1] + gamma [1] + 
        # + shift kernel size [SHIFT_SIZE] + erase vector [MEMORY_CONTENT_BITS] + write vector[MEMORY_BITS]  
        num_write_params = 3*self.num_memory_content_bits +1 +1 +1 +self.interface_shift_size
        
        # Write parameters - used during slicing.
        self.write_param_locations = self.calculate_param_locations({'query_vector': self.num_memory_content_bits, 'beta': 1, 'gate': 1,  
            'shift': self.interface_shift_size, 'gamma': 1, 
            'erase_vector': self.num_memory_content_bits, 'add_vector': self.num_memory_content_bits}, "Write")
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

    def forward(self, ctrl_hidden_state_BxH,  prev_memory_BxAxC,  prev_state_tuple):
        """
        Controller forward function. 
        
        :param ctrl_hidden_state_BxH: a Tensor with controller hidden state of size [BATCH_SIZE  x HIDDEN_SIZE]
        :param prev_memory_BxAxC: Previous state of the memory [BATCH_SIZE x  MEMORY_ADDRESSES x CONTENT_BITS] 
        :param prev_state_tuple: Tuple containing previous read and write attention vectors.
        :returns: Read vector, updated memory and state tuple (object of LSTMStateTuple class).
        """
        # Unpack previous cell  state - just to make sure that everything is ok...
        (prev_read_attentions_BxAx1_H,  prev_write_attention) = prev_state_tuple
        
        # !! Execute single step !!
        
        # Read attentions 
        read_attentions_BxAx1_H = []
        read_vectors_Bx1xC_H = []

        # Read heads.
        for i in range(self.interface_num_read_heads):
            # Calculate parameters of a given read head.
            params_BxP = self.hidden2read_list[i](ctrl_hidden_state_BxH)

            # Split the parameters.
            query_vector_BxC,  beta_Bx1,  gate_Bx1, shift_BxS, gamma_Bx1 = self.split_params(params_BxP,  self.read_param_locations)

            # Update the attention of a given head.
            attention_BxAx1 = self.update_attention(query_vector_BxC,  beta_Bx1,  gate_Bx1, shift_BxS, gamma_Bx1,  prev_memory_BxAxC,  prev_read_attentions_BxAx1_H[i])
            logger.debug("attention_BxAx1 {}:\n {}".format(attention_BxAx1.size(),  attention_BxAx1))  

            # Read vector from memory [BATCH_SIZE x CONTENT_BITS x 1].
            read_vector_Bx1xC = torch.matmul(torch.transpose(attention_BxAx1,  1, 2),  prev_memory_BxAxC)
            logger.debug("read_vector_Bx1xC {}:\n {}".format(read_vector_Bx1xC.size(),  read_vector_Bx1xC))  
            
            # Save read attentions and vectors.
            read_attentions_BxAx1_H.append(attention_BxAx1)
            read_vectors_Bx1xC_H.append(read_vector_Bx1xC)
            
        # Write head operation.
        # TODO
          
        # TODO:  REMOVE THOSE LINES.
        write_attention = prev_write_attention
        memory_BxAxC = prev_memory_BxAxC
        
        # Pack current cell state.
        state_tuple = InterfaceStateTuple(read_attentions_BxAx1_H,  write_attention)
        
        # Return read vector, new memory state and state tuple.
        return 0, memory_BxAxC,  state_tuple
 
    def calculate_param_locations(self,  param_sizes_dict,  head_name):
        """ Calculates locations of parameters, that will subsequently be used during parameter splitting.
        :param param_sizes_dict: Dictionary containing parameters along with their sizes (in bits/units).
        :param head_name: Name of head.
        :returns: "Locations" of parameters.
        """
        logger.debug("{} param sizes dict:\n {}".format(head_name, param_sizes_dict))        
        # Create the parameter lengths and store their cumulative sum
        lengths = np.fromiter(param_sizes_dict.values(), dtype=int)
        # Store "parameter locations" for further usage.
        param_locations = np.cumsum(np.insert(lengths, 0, 0), dtype=int).tolist()
        logger.debug("{} param locations:\n {}".format(head_name, param_locations))          
        return param_locations
        
    def split_params(self,  params,  locations):
        """ Split parameters into list on the basis of locations."""
        param_splits = [params[..., locations[i]:locations[i+1]]  for i in range(len(locations)-1)]
        logger.debug("Splitted params:\n {}".format(param_splits)) 
        return param_splits

    def update_attention(self,  query_vector_BxC,  beta_Bx1,  gate_Bx1, shift_BxS, gamma_Bx1,  prev_memory_BxAxC,  prev_attention_BxAx1):
        """ Updates the attention weights.
        
        :param query_vector_BxC: Query used for similarity calculation in content-based addressing [BATCH_SIZE x CONTENT_BITS]
        :param beta_Bx1: Strength parameter used in content-based addressing.
        :param gate_Bx1:
        :param shift_BxS:
        :param gamma_Bx1:
        :param prev_memory_BxAxC: tensor containing memory before update [BATCH_SIZE x MEMORY_ADDRESSES x CONTENT_BITS]
        :param prev_attention_BxAx1: previous attention vector [BATCH_SIZE x MEMORY_ADDRESSES x 1]
        :returns: attention vector of size [BATCH_SIZE x ADDRESS_SIZE x 1]
        """
        # Add 3rd dimensions where required and apply non-linear transformations.
        query_vector_Bx1xC = F.sigmoid(query_vector_BxC).unsqueeze(1) # I didn't have that non-linear transformation in TF!
        beta_Bx1x1 = F.softplus(beta_Bx1).unsqueeze(1)
        gate_Bx1x1 = F.sigmoid(gate_Bx1).unsqueeze(1)
        #gamma_Bx1x1 = tf.nn.softplus(params[:, self.slot_size+2:self.slot_size+3], name='gamma')
        #shift_BxSx1 = tf.nn.softmax(params[:, self.slot_size+3:], name='shift')
        
        # Content-based addressing.
        attention_Bx1xA = self.focusing_by_content(query_vector_Bx1xC,  beta_Bx1x1,  prev_memory_BxAxC)
    
        # Gating mechanism - choose beetween new attention from CBA or attention from previous iteration.

        # Location-based addressing.
        
        # TODO: rest;)
        
        return attention_Bx1xA
        
    def focusing_by_content(self,  query_vector_Bx1xC, beta_Bx1x1, prev_memory_BxAxC):
        """Computes content-based addressing. Uses query vectors for calculation of similarity.
        
        :param query_vector_Bx1xC: NTM "key"  [BATCH_SIZE x 1 x CONTENT_BITS] 
        :param beta_Bx1x1: key strength [BATCH_SIZE x 1 x 1]
        :param prev_memory_BxAxC: tensor containing memory before update [BATCH_SIZE x MEMORY_ADDRESSES x CONTENT_BITS]
        :returns: attention of size [BATCH_SIZE x ADDRESS_SIZE x 1]
        """
        # Normalize query batch - along samples.
        norm_query_vector_Bx1xC = F.normalize(query_vector_Bx1xC, p=2,  dim=2)
        logger.debug("norm_query_vector_Bx1xC {}:\n {}".format(norm_query_vector_Bx1xC.size(),  norm_query_vector_Bx1xC))  

        # Normalize memory - along addresses. 
        norm_memory_BxAxC = F.normalize(prev_memory_BxAxC, p=2,  dim=2)
        logger.debug("norm_memory_BxAxC {}:\n {}".format(norm_memory_BxAxC.size(),  norm_memory_BxAxC))  
        
        # Calculate cosine similarity [BATCH_SIZE x MEMORY_ADDRESSES x 1].
        similarity_BxAx1 = torch.matmul(norm_memory_BxAxC,  torch.transpose(norm_query_vector_Bx1xC,  1, 2))
        logger.debug("similarity_BxAx1 {}:\n {}".format(similarity_BxAx1.size(),  similarity_BxAx1))    
        
        # Element-wise multiplication [BATCH_SIZE x MEMORY_ADDRESSES x 1]
        strengthtened_similarity_BxAx1 = torch.matmul(similarity_BxAx1,  beta_Bx1x1)
        logger.debug("strengthtened_similarity_BxAx1 {}:\n {}".format(strengthtened_similarity_BxAx1.size(),  strengthtened_similarity_BxAx1))    

        # Calculate attention based on similarity along the "slot dimension" [BATCH_SIZE x MEMORY_ADDRESSES x 1].
        attention_BxAx1 = F.softmax(strengthtened_similarity_BxAx1, dim=1)
        logger.debug("attention_BxAx1 {}:\n {}".format(attention_BxAx1.size(),  attention_BxAx1))    
        return attention_BxAx1

