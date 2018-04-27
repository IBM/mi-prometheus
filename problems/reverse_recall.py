#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reverse_recall.py: Reversel recall problem"""
__author__      = "Tomasz Kornuta"

import numpy as np
import torch
from torch.autograd import Variable
from algorithmic_sequential_problem import AlgorithmicSequentialProblem

@AlgorithmicSequentialProblem.register
class ReverseRecallProblem(AlgorithmicSequentialProblem):
    """   
    Class generating sequences of random bit-patterns and targets forcing the system to learn sevese recall problem (a.k.a. reverse copy task).
    The formulation follows the original copy task from NTM paper, where:
    1) There are two markers, indicating
    - beginning of storing/memorization and
    - beginning of recalling from memory.
    2) For other elements of the sequence the command bits are set to zero
    3) Minor modification I: the target contains only data bits (command bits are skipped)
    4) Minor modification II: generator returns a mask, which can be used for filtering important elements of the output.
    
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
        self.dtype = torch.FloatTensor

    def generate_bit_sequence(self,  seq_length):
        """
        Generates a random sequence of random bit patterns.

        :param seq_length: the length of the sequence to be generated.
        :returns: Sequence of bit patterns [BATCH_SIZE x SEQ_LENGTH X DATA_BITS]
        """
        return np.random.binomial(1, self.bias, (self.batch_size, seq_length, self.data_bits))

    def generate_batch(self):
        """Generates a batch  of size [BATCH_SIZE, 2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS].
        Additional elements of sequence are  start and stop control markers, stored in additional bits.
       
        : returns: Tuple consisting of: input [BATCH_SIZE, 2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS], 
        output [BATCH_SIZE, 2*SEQ_LENGTH+2, DATA_BITS],
        mask [BATCH_SIZE, 2*SEQ_LENGTH+2]

        TODO: every item in batch has now the same seq_length.
        """
        # Set sequence length
        seq_length = np.random.randint(self.min_sequence_length, self.max_sequence_length+1)

        # Generate batch of random bit sequences.
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
        # Set bit sequence - but reversed.
        targets[:, seq_length+2:,  :] = np.fliplr(bit_seq)

        # Generate target mask: [BATCH_SIZE, 2*SEQ_LENGTH+2]
        mask = np.zeros([self.batch_size, 2*seq_length + 2])
        mask[:, seq_length+2:] = 1

        # PyTorch variables.
        ptinputs = Variable(torch.from_numpy(inputs).type(self.dtype))
        pttargets = Variable(torch.from_numpy(targets).type(self.dtype))
        ptmask = torch.from_numpy(mask).type(torch.uint8)


        # Return batch.
        return ptinputs,  pttargets,  ptmask

if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""
    
    # "Loaded parameters".
    params = {'name': 'reverse_recall', 'control_bits': 2, 'data_bits': 8, 'batch_size': 1, 'min_sequence_length': 1, 'max_sequence_length': 10,  'bias': 0.5}
    # Create problem object.
    problem = ReverseRecallProblem(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    (x, y, mask) = next(generator)
    # Display single sample (0) from batch.
    problem.show_sample(x, y, mask)
