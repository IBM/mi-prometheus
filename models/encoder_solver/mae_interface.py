#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mae_interface.py: pytorch module implementing MAE interface to the external memory."""
__author__ = "Tomasz Kornuta"


import torch 
import torch.nn.functional as F
import collections
import numpy as np
import logging
logger = logging.getLogger('NTM-Interface')

# Add path to main project directory.
import os, sys
from misc.app_state import AppState
sys.path.append(os.path.join(os.path.dirname(__file__),  '..', '..')) 


# Helper collection type.
_MAEInterfaceStateTuple = collections.namedtuple('MAEInterfaceStateTuple', ('attention',  'shift'))

class MAEInterfaceStateTuple(_MAEInterfaceStateTuple):
    """Tuple used by interface for storing current/past MAE interface state information"""
    __slots__ = ()


class MAEInterface(torch.nn.Module):
    """Class realizing interface between controller and memory in Memory Augmented Encoder cell.
    """
    def __init__(self, params):
        """ Constructor.
        
        :param params: Dictionary of parameters.
        """
        # Call constructor of base class.
        super(MAEInterface, self).__init__() 

        # Parse parameters.
        # Get hidden state size.
        self.ctrl_hidden_state_size = params['controller']['hidden_state_size']
        # Get memory parameters.
        self.num_memory_content_bits = params['memory']['num_content_bits']

        # Get interface parameters.
        self.interface_shift_size = params['interface']['shift_size']
        assert self.interface_shift_size % 2 != 0,  'Shift size must be an odd number'
        assert self.interface_shift_size >0,  'Shift size must be > 0'
 
        # -------------- WRITE HEAD -----------------#
        # Number/size of wrrite parameters: 
        # gamma [1] + shift kernel size [SHIFT_SIZE] + erase vector [MEMORY_CONTENT_BITS] + write vector[MEMORY_BITS]  
        num_write_params = 2*self.num_memory_content_bits +1 +self.interface_shift_size
        
        # Write parameters - used during slicing.
        self.write_param_locations = self.calculate_param_locations({  
            'shift': self.interface_shift_size, 'gamma': 1, 
            'erase_vector': self.num_memory_content_bits, 'add_vector': self.num_memory_content_bits}, "Write")
        assert num_write_params == self.write_param_locations[-1], "Last location must be equal to number of write params."
        
        # Forward linear layer that generates parameters of write head.
        self.hidden2write_params = torch.nn.Linear(self.ctrl_hidden_state_size,  num_write_params)

    def init_state(self,  batch_size,  num_memory_addresses):
        """
        Returns 'zero' (initial) state tuple.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :param num_memory_addresses: Number of memory addresses.
        :returns: Initial state tuple - object of InterfaceStateTuple class.
        """
        # Get dtype.
        dtype = AppState().dtype

        # Zero-hard attention.
        zh_attention = torch.zeros(batch_size, num_memory_addresses,  1).type(dtype)
        zh_attention[:, 0, 0] = 1
        # Init gating.
        init_shift = torch.zeros(batch_size, self.interface_shift_size,  1).type(dtype)
        init_shift[:, 1, 0] = 1
        
        # Return tuple.
        return MAEInterfaceStateTuple(zh_attention, init_shift)

    def forward(self, ctrl_hidden_state_BxH,  prev_memory_BxAxC,  prev_interface_state_tuple):
        """
        Controller forward function. 
        
        :param ctrl_hidden_state_BxH: a Tensor with controller hidden state of size [BATCH_SIZE  x HIDDEN_SIZE]
        :param prev_memory_BxAxC: Previous state of the memory [BATCH_SIZE x  MEMORY_ADDRESSES x CONTENT_BITS] 
        :param prev_interface_state_tuple: Tuple containing previous interface tuple.
        :returns: updated memory and state tuple (object of MAEInterfaceStateTuple class).
        """
        # Unpack previous cell state.
        (prev_write_attention_BxAx1, _) = prev_interface_state_tuple
         
        # !! Execute single step !!
            
        # Write head operation.
        # Calculate parameters of a given read head.
        params_BxP = self.hidden2write_params(ctrl_hidden_state_BxH)

        # Split the parameters.
        shift_BxS, gamma_Bx1,  erase_vector_BxC,  add_vector_BxC  = self.split_params(params_BxP,  self.write_param_locations)

        # Add 3rd dimensions where required and apply non-linear transformations.
        erase_vector_Bx1xC = F.sigmoid(erase_vector_BxC).unsqueeze(1) 
        add_vector_Bx1xC = F.sigmoid(add_vector_BxC).unsqueeze(1) 

        # Update the attention of the write head.
        write_attention_BxAx1,  write_state_tuple = self.update_attention(shift_BxS, gamma_Bx1,  prev_memory_BxAxC,  prev_write_attention_BxAx1)
        #logger.debug("write_attention_BxAx1 {}:\n {}".format(write_attention_BxAx1.size(),  write_attention_BxAx1))  

        # Update the memory.
        memory_BxAxC = self.update_memory(write_attention_BxAx1,  erase_vector_Bx1xC,  add_vector_Bx1xC,  prev_memory_BxAxC)
        
        # Return new memory state and state tuple.
        return memory_BxAxC,  write_state_tuple
 
    def calculate_param_locations(self,  param_sizes_dict,  head_name):
        """ Calculates locations of parameters, that will subsequently be used during parameter splitting.
        :param param_sizes_dict: Dictionary containing parameters along with their sizes (in bits/units).
        :param head_name: Name of head.
        :returns: "Locations" of parameters.
        """
        #logger.debug("{} param sizes dict:\n {}".format(head_name, param_sizes_dict))        
        # Create the parameter lengths and store their cumulative sum
        lengths = np.fromiter(param_sizes_dict.values(), dtype=int)
        # Store "parameter locations" for further usage.
        param_locations = np.cumsum(np.insert(lengths, 0, 0), dtype=int).tolist()
        #logger.debug("{} param locations:\n {}".format(head_name, param_locations))          
        return param_locations
        
    def split_params(self,  params,  locations):
        """ Split parameters into list on the basis of locations."""
        param_splits = [params[..., locations[i]:locations[i+1]]  for i in range(len(locations)-1)]
        #logger.debug("Splitted params:\n {}".format(param_splits)) 
        return param_splits


    def update_attention(self,  shift_BxS, gamma_Bx1,  prev_memory_BxAxC,  prev_attention_BxAx1):
        """ Updates the attention weights.
        
        :param shift_BxS: Convolution shift
        :param gamma_Bx1: Sharpening factor
        :param prev_memory_BxAxC: tensor containing memory before update [BATCH_SIZE x MEMORY_ADDRESSES x CONTENT_BITS]
        :param prev_attention_BxAx1: previous attention vector [BATCH_SIZE x MEMORY_ADDRESSES x 1]
        :returns: attention vector of size [BATCH_SIZE x ADDRESS_SIZE x 1]
        """
        # Add 3rd dimensions where required and apply non-linear transformations.
        # Produce location-addressing params.
        shift_BxSx1 = F.softmax(shift_BxS, dim=1).unsqueeze(2)
        # Gamma - oneplus.
        gamma_Bx1x1 =F.softplus(gamma_Bx1).unsqueeze(2) +1

        # Location-based addressing.
        location_attention_BxAx1 = self.location_based_addressing(prev_attention_BxAx1,  shift_BxSx1,  gamma_Bx1x1,  prev_memory_BxAxC)
        #logger.debug("location_attention_BxAx1 {}:\n {}".format(location_attention_BxAx1.size(),  location_attention_BxAx1))    
        
        return location_attention_BxAx1,  MAEInterfaceStateTuple(location_attention_BxAx1, shift_BxSx1)
        

    def location_based_addressing(self,  attention_BxAx1,  shift_BxSx1,  gamma_Bx1x1,  prev_memory_BxAxC):
        """ Computes location-based addressing, i.e. shitfts the head and sharpens.
        
        :param attention_BxAx1: Current attention [BATCH_SIZE x ADDRESS_SIZE x 1]
        :param shift_BxSx1: soft shift maks (convolutional kernel) [BATCH_SIZE x SHIFT_SIZE x 1]
        :param gamma_Bx1x1: sharpening factor [BATCH_SIZE x 1 x 1]
        :param prev_memory_BxAxC: tensor containing memory before update [BATCH_SIZE x MEMORY_ADDRESSES x CONTENT_BITS]
        :returns: attention vector of size [BATCH_SIZE x ADDRESS_SIZE x 1]
        """

        # 1. Perform circular convolution.
        shifted_attention_BxAx1 = self.circular_convolution(attention_BxAx1,  shift_BxSx1,  prev_memory_BxAxC)
        
        # 2. Perform Sharpening.
        sharpened_attention_BxAx1 = self.sharpening(shifted_attention_BxAx1,  gamma_Bx1x1)
               
        return sharpened_attention_BxAx1

    def circular_convolution(self,  attention_BxAx1,  shift_BxSx1,  prev_memory_BxAxC):
        """ Performs circular convoution, i.e. shitfts the attention accodring to given shift vector (convolution mask).
        
        :param attention_BxAx1: Current attention [BATCH_SIZE x ADDRESS_SIZE x 1]
        :param shift_BxSx1: soft shift maks (convolutional kernel) [BATCH_SIZE x SHIFT_SIZE x 1]
        :param prev_memory_BxAxC: tensor containing memory before update [BATCH_SIZE x MEMORY_ADDRESSES x CONTENT_BITS]
        :returns: attention vector of size [BATCH_SIZE x ADDRESS_SIZE x 1]
        """
        def circular_index(idx, num_addr):
            """ Calculates the index, taking into consideration the number of addresses in memory.
            :param idx: index (single element)
            :param num_addr: number of addresses in memory
            """
            if idx < 0: return num_addr + idx
            elif idx >= num_addr : return idx - num_addr
            else: return idx

        # Check whether inputs are already on GPU or not.
        long_dtype = AppState().LongTensor

        # Get number of memory addresses and batch size.
        batch_size =prev_memory_BxAxC.size(0) 
        num_addr = prev_memory_BxAxC.size(1)
        shift_size = self.interface_shift_size
        
        #logger.debug("shift_BxSx1 {}: {}".format(shift_BxSx1,  shift_BxSx1.size()))    
        # Create an extended list of indices indicating what elements of the sequence will be where.
        ext_indices_tensor = torch.Tensor([circular_index(shift, num_addr) for shift in range(-shift_size//2+1,  num_addr+shift_size//2)]).type(long_dtype)
        #logger.debug("ext_indices {}:\n {}".format(ext_indices_tensor.size(),  ext_indices_tensor))
    
        # Use indices for creation of an extended attention vector.
        ext_attention_BxEAx1 = torch.index_select(attention_BxAx1,  dim=1,  index=ext_indices_tensor)
        #logger.debug("ext_attention_BxEAx1 {}:\n {}".format(ext_attention_BxEAx1.size(),  ext_attention_BxEAx1))    
        
        # Transpose inputs to convolution.
        ext_att_trans_Bx1xEA = torch.transpose(ext_attention_BxEAx1,  1, 2)
        shift_trans_Bx1xS = torch.transpose(shift_BxSx1,  1,  2)
        # Perform  convolution for every batch-filter pair.
        tmp_attention_list = []
        for b in range(batch_size):
            tmp_attention_list.append(F.conv1d(ext_att_trans_Bx1xEA.narrow(0, b, 1),  shift_trans_Bx1xS.narrow(0, b, 1)))
        # Concatenate list into a single tensor.
        shifted_attention_BxAx1 = torch.transpose(torch.cat(tmp_attention_list,  dim=0),  1,  2)
        #logger.debug("shifted_attention_BxAx1 {}:\n {}".format(shifted_attention_BxAx1.size(),  shifted_attention_BxAx1))
        
        return shifted_attention_BxAx1

    def sharpening(self,  attention_BxAx1,  gamma_Bx1x1):
        """ Performs attention sharpening.
        
        :param attention_BxAx1: Current attention [BATCH_SIZE x ADDRESS_SIZE x 1]
        :param gamma_Bx1x1: sharpening factor [BATCH_SIZE x 1 x 1]
        :returns: attention vector of size [BATCH_SIZE x ADDRESS_SIZE x 1]
        """
                    
        # Power.        
        pow_attention_BxAx1 = torch.pow(attention_BxAx1,  gamma_Bx1x1)
        #logger.debug("pow_attention_BxAx1 {}:\n {}".format(pow_attention_BxAx1.size(),  pow_attention_BxAx1))
        
        # Normalize along addresses. 
        norm_attention_BxAx1 = F.normalize(pow_attention_BxAx1, p=1,  dim=1)
        #logger.debug("norm_attention_BxAx1 {}:\n {}".format(norm_attention_BxAx1.size(),  norm_attention_BxAx1))
  
        return norm_attention_BxAx1
        
    def update_memory(self,  write_attention_BxAx1,  erase_vector_Bx1xC,  add_vector_Bx1xC,  prev_memory_BxAxC):
        """ Returns 3D tensor of size [BATCH_SIZE x MEMORY_ADDRESSES x CONTENT_BITS] storing new content of the memory.
        
        :param write_attention_BxAx1: Current write attention [BATCH_SIZE x ADDRESS_SIZE x 1]
        :param erase_vector_Bx1xC: Erase vector [BATCH_SIZE x  1 x CONTENT_BITS]
        :param add_vector_Bx1xC: Add vector [BATCH_SIZE x 1 x CONTENT_BITS]
        :param prev_memory_BxAxC: tensor containing previous state of the memory [BATCH_SIZE x MEMORY_ADDRESSES x CONTENT_BITS]
        :returns: vector read from the memory [BATCH_SIZE x CONTENT_BITS]
        """
        # 1. Calculate the preserved content.
        preserve_content_BxAxC = 1 - torch.matmul(write_attention_BxAx1,  erase_vector_Bx1xC)
        # 2. Calculate the added content.
        add_content_BxAxC = torch.matmul(write_attention_BxAx1,  add_vector_Bx1xC) 
        # 3. Update memory.
        memory_BxAxC =  prev_memory_BxAxC * preserve_content_BxAxC + add_content_BxAxC  
        
        return memory_BxAxC
