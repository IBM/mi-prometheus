#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mas_interface.py: contains pytorch module implementing Memory Augmented Solver interface to external memory."""
__author__ = "Tomasz Kornuta"


import torch 
import torch.nn.functional as F
import collections
import numpy as np
import logging
# Set logging level.
logger = logging.getLogger('MAS-Interface')
#logging.basicConfig(level=logging.DEBUG)

from misc.app_state import AppState


# Helper collection type.
_MASInterfaceStateTuple = collections.namedtuple('MASInterfaceStateTuple', ('attention', 'final_encoder_attention', 'gate', 'shift'))

class MASInterfaceStateTuple(_MASInterfaceStateTuple):
    """Tuple used by interface for storing current/past Memory Augmented Solver interface state information"""
    __slots__ = ()


class MASInterface(torch.nn.Module):
    """Class realizing interface between MAS controller and memory.
    """
    def __init__(self, params):
        """ Constructor.
        
        :param params: Dictionary of parameters.
        """
        # Call constructor of base class.
        super(MASInterface, self).__init__() 

        # Parse parameters.
        # Get hidden state size.
        self.ctrl_hidden_state_size = params['controller']['hidden_state_size']
        # Get memory parameters.
        self.num_memory_content_bits = params['memory']['num_content_bits']

        # Get interface parameters.
        self.interface_shift_size = params['mas_interface']['shift_size']
        assert self.interface_shift_size % 2 != 0,  'Shift size must be an odd number'
        assert self.interface_shift_size >0,  'Shift size must be > 0'

        # -------------- READ HEAD -----------------#
        # Number/size of parameters of a read head: gate [3] + gamma [1] + shift kernel size [SHIFT_SIZE]
        num_read_params =  (3 +1 +self.interface_shift_size)
        # Dictionary with read parameters - used during slicing.
        self.read_param_locations = self.calculate_param_locations({'gate': 3,  
            'shift': self.interface_shift_size, 'gamma': 1},  "Read")
        assert num_read_params == self.read_param_locations[-1], "Last location must be equal to number of read params."

        # Forward linear layer that generates parameters of read head.
        self.hidden2read_params = torch.nn.Linear(self.ctrl_hidden_state_size,  num_read_params)

    def init_state(self,  batch_size,  num_memory_addresses, final_encoder_attention_BxAx1):
        """
        Returns 'zero' (initial) state tuple.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :param num_memory_addresses: Number of memory addresses.
        :param final_encoder_attention_BxAx1: final attention of the encoder [BATCH_SIZE x MEMORY_ADDRESSES x 1]
        :returns: Initial state tuple - object of InterfaceStateTuple class.
        """
        # Get dtype.
        dtype = AppState().dtype

        # Initial  attention weights [BATCH_SIZE x MEMORY_ADDRESSES x 1]
        # Zero-hard attention.
        zh_attention = torch.zeros(batch_size, num_memory_addresses,  1).type(dtype)
        zh_attention[:, 0, 0] = 1 # Initialize as "hard attention on 0 address"

        # Gating [BATCH x 3 x 1]
        init_gating = torch.zeros(batch_size, 3,  1).type(dtype)
        init_gating[:, 0, 0] = 1 # Initialize as "prev attention"

        # Shift [BATCH x SHIFT_SIZE x 1]
        init_shift = torch.zeros(batch_size, self.interface_shift_size,  1).type(dtype)
        init_shift[:, 1, 0] = 1 # Initialize as "0 shift".

        # Remember zero-hard attention.        
        self.zero_hard_attention_BxAx1 = zh_attention
        # Remember final attention of encoder.
        self.final_encoder_attention_BxAx1 = final_encoder_attention_BxAx1

        # Return tuple.
        return MASInterfaceStateTuple(zh_attention, self.final_encoder_attention_BxAx1, init_gating, init_shift)

    def forward(self, ctrl_hidden_state_BxH,  prev_memory_BxAxC,  prev_interface_state_tuple):
        """
        Controller forward function. 
        
        :param ctrl_hidden_state_BxH: a Tensor with controller hidden state of size [BATCH_SIZE  x HIDDEN_SIZE]
        :param prev_memory_BxAxC: Previous state of the memory [BATCH_SIZE x  MEMORY_ADDRESSES x CONTENT_BITS] 
        :param prev_interface_state_tuple: Tuple containing previous read and write attention vectors.
        :returns: List of read vectors [BATCH_SIZE x CONTENT_SIZE], updated memory and state tuple (object of LSTMStateTuple class).
        """
       # Unpack cell state.
        (prev_read_attention_BxAxH, _, _, _) = prev_interface_state_tuple
         
        # !! Execute single step !!

        # Calculate parameters of a read read head.
        params_BxP = self.hidden2read_params(ctrl_hidden_state_BxH)

        # Split the parameters.
        gate_Bx3, shift_BxS, gamma_Bx1 = self.split_params(params_BxP,  self.read_param_locations)

        # Update the attention of a given read head.
        read_attention_BxAx1,  interface_state_tuple = self.update_attention(gate_Bx3, shift_BxS, gamma_Bx1,  prev_memory_BxAxC,  prev_read_attention_BxAxH)
        #logger.debug("read_attention_BxAx1 {}:\n {}".format(read_attention_BxAx1.size(),  read_attention_BxAx1))  

        # Read vector from memory [BATCH_SIZE x CONTENT_BITS].
        read_vector_BxC = self.read_from_memory(read_attention_BxAx1,  prev_memory_BxAxC)


        # Return read vector, prev memory state and state tuple.
        return read_vector_BxC, prev_memory_BxAxC,  interface_state_tuple
 
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

    def update_attention(self,  gate_Bx3, shift_BxS, gamma_Bx1,  prev_memory_BxAxC,  prev_attention_BxAx1):
        """ Updates the attention weights.
        
        :param gate_Bx3:
        :param shift_BxS:
        :param gamma_Bx1:
        :param prev_memory_BxAxC: tensor containing memory before update [BATCH_SIZE x MEMORY_ADDRESSES x CONTENT_BITS]
        :param prev_attention_BxAx1: previous attention vector [BATCH_SIZE x MEMORY_ADDRESSES x 1]
        :returns: attention vector of size [BATCH_SIZE x ADDRESS_SIZE x 1]
        """
        # Add 3rd dimensions where required and apply non-linear transformations.


        # Produce gating param.
        gate_Bx3x1 = F.sigmoid(gate_Bx3).unsqueeze(2)
        gate0_Bx3x1 = gate_Bx3x1[:,0,:].unsqueeze(2)
        gate1_Bx3x1 = gate_Bx3x1[:,1,:].unsqueeze(2)
        gate2_Bx3x1 = gate_Bx3x1[:,2,:].unsqueeze(2)
        #logger.debug("gate0_Bx2x1 {}:\n {}".format(gate0_Bx2x1.size(), gate0_Bx2x1))
    
        # Produce location-addressing params.
        shift_BxSx1 = F.softmax(shift_BxS, dim=1).unsqueeze(2)
        # Gamma - oneplus.
        gamma_Bx1x1 =F.softplus(gamma_Bx1).unsqueeze(2) +1
    
        # Gating mechanism - choose beetween new attention from CBA or attention from previous iteration. [BATCH_SIZE x ADDRESSES x 1].
        #logger.debug("prev_attention_BxAx1 {}:\n {}".format(prev_attention_BxAx1.size(),  prev_attention_BxAx1))    

        # Location-based addressing.
        location_attention_BxAx1 = self.location_based_addressing(prev_attention_BxAx1,  shift_BxSx1,  gamma_Bx1x1,  prev_memory_BxAxC)
        #logger.debug("location_attention_BxAx1 {}:\n {}".format(location_attention_BxAx1.size(),  location_attention_BxAx1))   

    
        attention_after_gating_BxAx1 = gate0_Bx3x1 * location_attention_BxAx1  + \
            gate1_Bx3x1 * self.final_encoder_attention_BxAx1 + \
            gate2_Bx3x1 * self.zero_hard_attention_BxAx1

        #logger.debug("attention_after_gating_BxAx1 {}:\n {}".format(attention_after_gating_BxAx1.size(),  attention_after_gating_BxAx1))    
        int_state = MASInterfaceStateTuple(attention_after_gating_BxAx1, self.final_encoder_attention_BxAx1, gate_Bx3x1, shift_BxSx1)
        return attention_after_gating_BxAx1, int_state


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
        #dtype = torch.cuda.LongTensor if attention_BxAx1.is_cuda else torch.LongTensor
        dtype = AppState().LongTensor

        # Get number of memory addresses and batch size.
        batch_size =prev_memory_BxAxC.size(0) 
        num_addr = prev_memory_BxAxC.size(1)
        shift_size = self.interface_shift_size
        
        #logger.debug("shift_BxSx1 {}: {}".format(shift_BxSx1,  shift_BxSx1.size()))    
        # Create an extended list of indices indicating what elements of the sequence will be where.
        ext_indices_tensor = torch.Tensor([circular_index(shift, num_addr) for shift in range(-shift_size//2+1,  num_addr+shift_size//2)]).type(dtype)
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
        
        # Manual test of convolution
        #sum = 0
        #el =0
        #b = 0
        #for i in range(3):
        #    sum += ext_attention_BxEAx1[b][el+i][0] * shift_BxSx1[b][i][0]
        #print("SUM= ", sum)
        return shifted_attention_BxAx1

    def sharpening(self,  attention_BxAx1,  gamma_Bx1x1):
        """ Performs attention sharpening.
        
        :param attention_BxAx1: Current attention [BATCH_SIZE x ADDRESS_SIZE x 1]
        :param gamma_Bx1x1: sharpening factor [BATCH_SIZE x 1 x 1]
        :returns: attention vector of size [BATCH_SIZE x ADDRESS_SIZE x 1]
        """
        #gamma_Bx1x1[0][0][0]=40
        #gamma_Bx1x1[0][0][0]=10
        
        #logger.debug("gamma_Bx1x1 {}:\n {}".format(gamma_Bx1x1.size(),  gamma_Bx1x1))
                    
        # Power.        
        pow_attention_BxAx1 = torch.pow(attention_BxAx1 + 1e-12,  gamma_Bx1x1)
        #logger.debug("pow_attention_BxAx1 {}:\n {}".format(pow_attention_BxAx1.size(),  pow_attention_BxAx1))
        
        # Normalize along addresses. 
        norm_attention_BxAx1 = F.normalize(pow_attention_BxAx1, p=1,  dim=1)
        #logger.debug("norm_attention_BxAx1 {}:\n {}".format(norm_attention_BxAx1.size(),  norm_attention_BxAx1))
  
        return norm_attention_BxAx1


    def read_from_memory(self,  attention_BxAx1,  memory_BxAxC):
        """ Returns 2D tensor of size [BATCH_SIZE x CONTENT_BITS] storing vector read from memory given the attention.
        
        :param attention_BxAx1: Current attention [BATCH_SIZE x ADDRESS_SIZE x 1]
        :param memory_BxAxC: tensor containing memory [BATCH_SIZE x MEMORY_ADDRESSES x CONTENT_BITS]
        :returns: vector read from the memory [BATCH_SIZE x CONTENT_BITS]
        """
        read_vector_Bx1xC = torch.matmul(torch.transpose(attention_BxAx1,  1, 2),  memory_BxAxC)
        #logger.debug("read_vector_Bx1xC {}:\n {}".format(read_vector_Bx1xC.size(),  read_vector_Bx1xC))  

        # Return 2D tensor.
        return read_vector_Bx1xC.squeeze(dim=1)

