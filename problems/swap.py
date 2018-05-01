#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""serial_recall_original.py: Original serial recall problem (a.k.a. copy task)"""
__author__      = "Tomasz Kornuta"

import numpy as np
import torch
from torch.autograd import Variable
from algorithmic_sequential_problem import AlgorithmicSequentialProblem

@AlgorithmicSequentialProblem.register
class SwapProblem(AlgorithmicSequentialProblem):
    """   
    Creates input being a sequence of bit pattern and target being the same sequence "bitshifted" by num_items to right.
    For example:
    num_items = 2 -> seq_items >> 2 
    num_items = -1 -> seq_items << 1
    Offers two modes of operation, depending on the value of num_items parameter:
    1)  -1 < num_item < 1: relative mode, where num_item represents the % of length of the sequence by which it should be shifted
    2) otherwise: absolute number of items by which the sequence will be shifted.
    
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
        self.num_items = params['num_items']
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
        if seq_length % 2:
            seq_length = seq_length + 1

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
        # Set bit sequence.

        # Rotate sequence by shifting the items to right: seq >> num_items
        # i.e num_items = 2 -> seq_items >> 2 
        # and num_items = -1 -> seq_items << 1
        # For that reason we must change the sign of num_items
        num_items = -self.num_items
        # Check if we are using relative or absolute rotation.
        if -1 < num_items < 1:
            num_items = num_items * seq_length
        # Round items shift  to int.
        num_items = np.round(num_items)
        # Modulo items shift with length of the sequence.
        num_items = int(num_items % seq_length)

        # Apply items shift 
        bit_seq = np.concatenate((bit_seq[:, num_items:, :], bit_seq[:, :num_items, :]), axis=1)
        targets[:, seq_length+2:,  :] = bit_seq

        # Generate target mask: [BATCH_SIZE, 2*SEQ_LENGTH+2]
        targets_mask = torch.zeros([self.batch_size, 2*seq_length + 2]).type(torch.ByteTensor)
        targets_mask[:, seq_length+2:] = 1

        # PyTorch variables.
        ptinputs = Variable(torch.from_numpy(inputs).type(self.dtype))
        pttargets = Variable(torch.from_numpy(targets).type(self.dtype))

        # Return batch.
        return ptinputs,  pttargets,  targets_mask

if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""
    
    # "Loaded parameters".
    params = {'control_bits': 2, 'data_bits': 8, 'batch_size': 1, 
        'min_sequence_length': 1, 'max_sequence_length': 10,  'bias': 0.5, 'num_items':2}
    # Create problem object.
    problem = SwapProblem(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    (x, y, mask) = next(generator)
    # Display single sample (0) from batch.
    problem.show_sample(x, y, mask)
