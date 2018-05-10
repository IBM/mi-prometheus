#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""manipulation_spatial_rotate.py: Spatial rotation (bitshift) for all items in the sequence"""
__author__      = "Tomasz Kornuta"

import numpy as np
import torch
from torch.autograd import Variable
from algorithmic_sequential_problem import AlgorithmicSequentialProblem

@AlgorithmicSequentialProblem.register
class ManipulationSpatialRotation(AlgorithmicSequentialProblem):
    """   
    Creates input being a sequence of bit pattern and target being the same sequence, but with data_bits "bitshifted" by num_bits to right.
    Offers two modes of operation, depending on the value of num_bits parameter:
    1)  -1 < num_bits < 1: relative mode, where num_bits represents the % of data bits by which every should be shifted
    2) otherwise: absolute number of bits by which the sequence will be shifted.
    
    TODO: sequences of different lengths in batch (filling with zeros?)
    """
    def __init__(self,  params):
        """ 
        Constructor - stores parameters.
        
        :param params: Dictionary of parameters.
        """
        # Retrieve parameters from the dictionary.
        self.batch_size = params['batch_size']
        # Number of bits in one element.
        self.control_bits = params['control_bits']
        self.data_bits = params['data_bits']
        assert self.control_bits >=2, "Problem requires at least 2 control bits (currently %r)" % self.control_bits
        assert self.data_bits >=2, "Problem requires at least 1 data bit (currently %r)" % self.data_bits
        # Min and max lengts (number of elements).
        self.min_sequence_length = params['min_sequence_length']
        self.max_sequence_length = params['max_sequence_length']
        # Parameter  denoting 0-1 distribution (0.5 is equal).
        self.bias = params['bias']
        self.num_bits = params['num_bits']
        self.dtype = torch.FloatTensor

    def generate_batch(self):
        """Generates a batch  of size [BATCH_SIZE, 2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS].
        Additional elements of sequence are  start and stop control markers, stored in additional bits.
       
        : returns: Tuple consisting of: input [BATCH_SIZE, 2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS], 
        output [BATCH_SIZE, 2*SEQ_LENGTH+2, DATA_BITS],
        mask [BATCH_SIZE, 2*SEQ_LENGTH+2]

        TODO: every item in batch has now the same seq_length.
        """
        # Set sequence length.
        seq_length = np.random.randint(self.min_sequence_length, self.max_sequence_length+1)

        # Generate batch of random bit sequences [BATCH_SIZE x SEQ_LENGTH X DATA_BITS]
        bit_seq = np.random.binomial(1, self.bias, (self.batch_size, seq_length, self.data_bits))

        # Generate input:  [BATCH_SIZE, 2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS]
        inputs = np.zeros([self.batch_size, 2*seq_length + 2, self.control_bits +  self.data_bits], dtype=np.float32)
        # Set start control marker.
        inputs[:, 0, 0] = 1 # Memorization bit.
        # Set bit sequence.
        inputs[:, 1:seq_length+1,  self.control_bits:self.control_bits+self.data_bits] = bit_seq
        # Set end control marker.
        inputs[:, seq_length+1, 1] = 1 # Recall bit.
        
        # Generate target:  [BATCH_SIZE, 2*SEQ_LENGTH+2, DATA_BITS] (only data bits!)
        targets = np.zeros([self.batch_size, 2*seq_length + 2,  self.data_bits], dtype=np.float32)

        # Rotate sequence by shifting the bits to right: data_bits >> num_bits
        num_bits =-self.num_bits
        # Check if we are using relative or absolute rotation.
        if -1 < num_bits < 1:
            num_bits = num_bits * seq_length
        # Round bitshift  to int.
        num_bits = np.round(num_bits)
        # Modulo bitshift with data_bits.
        num_bits = int(num_bits % self.data_bits)

        # Apply items shift 
        bit_seq = np.concatenate((bit_seq[:, :, num_bits:], bit_seq[:, :, :num_bits]), axis=2)
        targets[:, seq_length+2:,  :] = bit_seq

        # Generate target mask: [BATCH_SIZE, 2*SEQ_LENGTH+2]
        targets_mask = torch.zeros([self.batch_size, 2*seq_length + 2]).type(torch.ByteTensor)
        targets_mask[:, seq_length+2:] = 1

        # PyTorch variables.
        ptinputs = torch.from_numpy(inputs).type(self.dtype)
        pttargets = torch.from_numpy(targets).type(self.dtype)

        # Return batch.
        return ptinputs,  pttargets,  targets_mask

    # method for changing the maximum length, used mainly during curriculum learning
    def set_max_length(self, max_length):
        self.max_sequence_length = max_length

if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""
    
    # "Loaded parameters".
    params = {'control_bits': 2, 'data_bits': 8, 'batch_size': 1, 
        'min_sequence_length': 1, 'max_sequence_length': 10,  'bias': 0.5, 'num_bits':0.5}
    # Create problem object.
    problem = ManipulationSpatialRotation(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    (x, y, mask) = next(generator)
    # Display single sample (0) from batch.
    problem.show_sample(x, y, mask)
