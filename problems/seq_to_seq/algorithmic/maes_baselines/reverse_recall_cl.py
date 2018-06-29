#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reverse_recall_cl.py: Contains definition of reverse recall problem with control markers and command lines"""
__author__      = "Tomasz Kornuta"

# Add path to main project directory - required for testing of the main function and see whether problem is working at all (!)
import os,  sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..','..','..','..')) 

import torch
import numpy as np
from problems.problem import DataTuple
from problems.seq_to_seq.algorithmic.algorithmic_seq_to_seq_problem import AlgorithmicSeqToSeqProblem, AlgSeqAuxTuple


class ReverseRecallCommandLines(AlgorithmicSeqToSeqProblem):
    """   
    Class generating sequences of random bit-patterns and targets forcing the system to learn reverse recall problem.
    1) There are two markers, indicatinn:
    - beginning of storing/memorization and
    - beginning of recalling from memory.
    2) Additionally, there is a command line (3rd command bit) indicating whether given item is to be stored in mememory (0) or recalled (1).
    """
    def __init__(self,  params):
        """ 
        Constructor - stores parameters. Calls parent class initialization.
        
        :param params: Dictionary of parameters.
        """
        # Call parent constructor - sets e.g. the loss function ;)
        super(ReverseRecallCommandLines, self).__init__(params)
        
        # Retrieve parameters from the dictionary.
        self.batch_size = params['batch_size']
        # Number of bits in one element.
        self.control_bits = params['control_bits']
        self.data_bits = params['data_bits']
        assert self.control_bits >=3, "Problem requires at least 3 control bits (currently %r)" % self.control_bits
        assert self.data_bits >=1, "Problem requires at least 1 data bit (currently %r)" % self.data_bits
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
        # Define control channel bits.
        # ctrl_main = [0, 0, 0] # not really used.
        ctrl_aux = [0, 0, 1]

        # Markers.
        marker_start_main = [1, 0, 0]
        marker_start_aux = [0, 1, 0]

        # Set sequence length.
        seq_length = np.random.randint(self.min_sequence_length, self.max_sequence_length+1)

        # Generate batch of random bit sequences [BATCH_SIZE x SEQ_LENGTH X DATA_BITS]
        bit_seq = np.random.binomial(1, self.bias, (self.batch_size, seq_length, self.data_bits))
        
        # 1. Generate inputs.
        # Generate input:  [BATCH_SIZE, 2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS]
        inputs = np.zeros([self.batch_size, 2*seq_length + 2, self.control_bits +  self.data_bits], dtype=np.float32)

        # Set start main control marker.
        inputs[:, 0, 0:3] = np.tile(marker_start_main, (self.batch_size, 1))

        # Set bit sequence.
        inputs[:, 1:seq_length+1,  self.control_bits:self.control_bits+self.data_bits] = bit_seq
        # inputs[:, 1:seq_length+1, 0:3] = np.tile(ctrl_main, (self.batch_size, seq_length,1)) # not used as ctrl_main is all zeros.

        # Set start aux control marker.
        inputs[:, seq_length+1, 0:3] = np.tile(marker_start_aux, (self.batch_size, 1))
        inputs[:, seq_length+2:2*seq_length+2, 0:3] = np.tile(ctrl_aux, (self.batch_size, seq_length,1))
        
        # 2. Generate targets.
        # Generate target:  [BATCH_SIZE, 2*SEQ_LENGTH+2, DATA_BITS] (only data bits!)
        targets = np.zeros([self.batch_size, 2*seq_length + 2,  self.data_bits], dtype=np.float32)
        # Set bit sequence.
        targets[:, seq_length+2:,  :] = np.fliplr(bit_seq)

        # 3. Generate mask.
        # Generate target mask: [BATCH_SIZE, 2*SEQ_LENGTH+2]
        mask = torch.zeros([self.batch_size, 2*seq_length + 2]).type(torch.ByteTensor)
        mask[:, seq_length+2:] = 1

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
    params = {'control_bits': 3, 'data_bits': 8, 'batch_size': 2, 
        'min_sequence_length': 1, 'max_sequence_length': 10,  'bias': 0.5}
    # Create problem object.
    problem = ReverseRecallCommandLines(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    data_tuple, aux_tuple = next(generator)
    # Display single sample (0) from batch.
    problem.show_sample(data_tuple, aux_tuple)
