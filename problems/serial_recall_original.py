#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""serial_recall_original.py: Original serial recall problem (a.k.a. copy task)"""
__author__      = "Tomasz Kornuta"

import numpy as np
import torch
from algorithmic_sequential_problem import AlgorithmicSequentialProblem
from algorithmic_sequential_problem import DataTuple

@AlgorithmicSequentialProblem.register
class SerialRecall(AlgorithmicSequentialProblem):
    """   
    Class generating sequences of random bit-patterns and targets forcing the system to learn serial recall problem (a.k.a. copy task).
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
        targets[:, seq_length+2:,  :] = bit_seq

        # Generate target mask: [BATCH_SIZE, 2*SEQ_LENGTH+2]
        mask = torch.zeros([self.batch_size, 2*seq_length + 2]).type(torch.ByteTensor)
        mask[:, seq_length+2:] = 1

        # PyTorch variables.
        ptinputs = torch.from_numpy(inputs).type(self.dtype)
        pttargets = torch.from_numpy(targets).type(self.dtype)

        # Return data tuple.
        return DataTuple(ptinputs, pttargets, mask)

    # method for changing the maximum length, used mainly during curriculum learning
    def set_max_length(self, max_length):
        self.max_sequence_length = max_length

if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""
    
    # "Loaded parameters".
    params = {'control_bits': 2, 'data_bits': 8, 'batch_size': 1, 
        'min_sequence_length': 1, 'max_sequence_length': 10,  'bias': 0.5}
    # Create problem object.
    problem = SerialRecall(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    (x, y, mask) = next(generator)
    # Display single sample (0) from batch.
    problem.show_sample(x, y, mask)
