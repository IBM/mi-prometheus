#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""serial_recall_simplified.py: Simplified serial recall problem (a.k.a. copy task)"""
__author__      = "Tomasz Kornuta"

# Add path to main project directory - required for testing of the main function and see whether problem is working at all (!)
import os,  sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..','..','..')) 

import torch
import numpy as np
from problems.problem import DataTuple
from problems.seq_to_seq.algorithmic.algorithmic_seq_to_seq_problem import AlgorithmicSeqToSeqProblem, AlgSeqAuxTuple


class SerialRecallSimplified(AlgorithmicSeqToSeqProblem):
    """   
    Class generating sequences of random bit-patterns and targets forcing the system to learn serial recall problem (a.k.a. copy task).
    Assumes several simplifications in comparison to copy task from NTM paper, i.e.:
    1) Major modification: there are no markers indicating beginning and of storing and recalling. Instead, is uses a single control bit to indicate whether this is item should be stored or recalled from memory.
    2) Minor modification I: the target contains only data bits (command bits are skipped)
    3) Minor modification II: generator returns a mask, which can be used for filtering important elements of the output.
    
    TODO: sequences of different lengths in batch (filling with zeros?)
    """
    def __init__(self,  params):
        """ 
        Constructor - stores parameters. Calls parent class initialization.
        
        :param params: Dictionary of parameters.
        """
        # Call parent constructor - sets e.g. the loss function ;)
        super(SerialRecallSimplified, self).__init__(params)
        
        # Retrieve parameters from the dictionary.
        self.batch_size = params['batch_size']
        # Number of bits in one element.
        self.control_bits = params['control_bits']
        self.data_bits = params['data_bits']
        assert self.control_bits >=1, "Problem requires at least 1 control bit (currently %r)" % self.control_bits
        assert self.data_bits >=1, "Problem requires at least 1 data bit (currently %r)" % self.data_bits
        # Min and max lengts (number of elements).
        self.min_sequence_length = params['min_sequence_length']
        self.max_sequence_length = params['max_sequence_length']
        # Parameter  denoting 0-1 distribution (0.5 is equal).
        self.bias = params['bias']
        self.dtype = torch.FloatTensor

    def generate_batch(self):
        """Generates a batch  of size [BATCH_SIZE, 2*SEQ_LENGTH, CONTROL_BITS+DATA_BITS].
        Additional elements of sequence are  start and stop control markers, stored in additional bits.
       
        :param seq_length: the length of the copy sequence.
        : returns: Tuple consisting of: input [BATCH_SIZE, 2*SEQ_LENGTH, CONTROL_BITS+DATA_BITS], 
        output [BATCH_SIZE, 2*SEQ_LENGTH, DATA_BITS],
        mask [BATCH_SIZE, 2*SEQ_LENGTH]

        TODO: every item in batch has now the same seq_length.
        """
        # Set sequence length.
        seq_length = np.random.randint(self.min_sequence_length, self.max_sequence_length+1)

        # Generate batch of random bit sequences [BATCH_SIZE x SEQ_LENGTH X DATA_BITS]
        bit_seq = np.random.binomial(1, self.bias, (self.batch_size, seq_length, self.data_bits))
        
        # Generate input:  [BATCH_SIZE, 2*SEQ_LENGTH, CONTROL_BITS+DATA_BITS]
        inputs = np.zeros([self.batch_size, 2*seq_length, self.control_bits +  self.data_bits], dtype=np.float32)
        # Set memorization bit for the whole bit sequence that need to be memorized.
        inputs[:, seq_length:, 0] = 1
        # Set bit sequence.
        inputs[:, :seq_length,  self.control_bits:self.control_bits+self.data_bits] = bit_seq
        
        # Generate target:  [BATCH_SIZE, 2*SEQ_LENGTH, DATA_BITS] (only data bits!)
        targets = np.zeros([self.batch_size, 2*seq_length,  self.data_bits], dtype=np.float32)
        # Set bit sequence.
        targets[:, seq_length:,  :] = bit_seq
        
        # Generate target mask: [BATCH_SIZE, 2*SEQ_LENGTH]
        mask = torch.zeros([self.batch_size, 2*seq_length]).type(torch.ByteTensor)
        mask[:, seq_length:] = 1

        # PyTorch variables.
        ptinputs = torch.from_numpy(inputs).type(self.dtype)
        pttargets = torch.from_numpy(targets).type(self.dtype)

        # Return tuples.
        data_tuple = DataTuple(ptinputs, pttargets)
        aux_tuple = AlgSeqAuxTuple(mask, seq_length, 1)

        return data_tuple, aux_tuple


    # method for changing the maximum length, used mainly during curriculum learning
    def set_max_length(self, max_length):
        self.max_sequence_length = max_length
        
if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""
    
    # "Loaded parameters".
    params = {'control_bits': 2, 'data_bits': 8, 'batch_size': 2,
              'min_sequence_length': 1, 'max_sequence_length': 10, 'bias': 0.5}
    # Create problem object.
    problem = SerialRecallSimplified(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    data_tuple,  aux_tuple = next(generator)
    # Display single sample (0) from batch.
    problem.show_sample(data_tuple, aux_tuple)

