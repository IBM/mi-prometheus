#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""repeat_reverse_recall_cl.py: Contains definition of  repeat reverse recall problem with control markers and command lines"""
__author__      = "Tomasz Kornuta"

# Add path to main project directory - required for testing of the main function and see whether problem is working at all (!)
import os,  sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..','..','..','..')) 

import torch
import numpy as np
from problems.problem import DataTuple
from problems.seq_to_seq.algorithmic.algorithmic_seq_to_seq_problem import AlgorithmicSeqToSeqProblem, AlgSeqAuxTuple


class RepeatReverseRecallCommandLines(AlgorithmicSeqToSeqProblem):
    """   
    Class generating sequences of random bit-patterns and targets forcing the system to learn repeated reverse recall problem.
    1) There are 2 markers, indicatinn:
    - beginning of storing/memorization,
    - beginning of forward recalling from memory,
    2) Additionally, there is a command line (3rd command bit) indicating whether given item is to be stored in mememory (0)
     or recalled (1).
    """
    def __init__(self,  params):
        """ 
        Constructor - stores parameters. Calls parent class initialization.
        
        :param params: Dictionary of parameters.
        """
        # Call parent constructor - sets e.g. the loss function ;)
        super(RepeatReverseRecallCommandLines, self).__init__(params)
        
        # Retrieve parameters from the dictionary.
        self.batch_size = params['batch_size']
        # Number of bits in one element.
        self.control_bits = params['control_bits']
        self.data_bits = params['data_bits']
        assert self.control_bits >=3, "Problem requires at least 3 control bits (currently %r)" % self.control_bits
        assert self.data_bits >=1, "Problem requires at least 1 data bit (currently %r)" % self.data_bits
        self.randomize_control_lines = params.get('randomize_control_lines', False)

        # Min and max lengts (number of elements).
        self.min_sequence_length = params['min_sequence_length']
        self.max_sequence_length = params['max_sequence_length']
        # Min and max number of recalls
        self.min_recall_number = params.get('min_recall_number', 1)
        self.max_recall_number = params.get('max_recall_number', 5)

        # Parameter  denoting 0-1 distribution (0.5 is equal).
        self.bias = params['bias']
        self.dtype = torch.FloatTensor

    def generate_batch(self):
        """Generates a batch  of size [BATCH_SIZE, 2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS].
        Additional elements of sequence are  start and stop control markers, stored in additional bits.
       
        : returns: Tuple consisting of: input [BATCH_SIZE, 2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS], 
        output [BATCH_SIZE, 2*SEQ_LENGTH+2, DATA_BITS],
        mask [BATCH_SIZE, 2*SEQ_LENGTH+2]

        """
        # Define control channel bits.
        # ctrl_main = [0, 0, 0] # not really used.

        #ctrl_aux[2:self.control_bits] = 1 #[0, 0, 1]
        ctrl_aux = np.zeros(self.control_bits)
        if (self.control_bits == 3):
            ctrl_aux[2] = 1 #[0, 0, 1]
        else:
            if self.randomize_control_lines:
                # Randomly pick one of the bits to be set.
                ctrl_bit = np.random.randint(2, self.control_bits)
                ctrl_aux[ctrl_bit] = 1
            else:
                ctrl_aux[self.control_bits-1] = 1


        # Markers.
        marker_start_main = np.zeros(self.control_bits)
        marker_start_main[0] = 1 #[1, 0, 0]
        marker_start_aux = np.zeros(self.control_bits)
        marker_start_aux[1] = 1 #[0, 1, 0]

        # Set sequence length.
        seq_length = np.random.randint(self.min_sequence_length, self.max_sequence_length+1)

        # Number of recalls.
        recall_number = np.random.randint(self.min_recall_number, self.max_recall_number+1)

        # Generate batch of random bit sequences [BATCH_SIZE x SEQ_LENGTH X DATA_BITS]
        bit_seq = np.random.binomial(1, self.bias, (self.batch_size, seq_length, self.data_bits))
        
        # 1. Generate inputs.
        # Generate input:  [BATCH_SIZE, 3*SEQ_LENGTH+3, CONTROL_BITS+DATA_BITS]
        inputs = np.zeros([self.batch_size, (recall_number+1)*(seq_length+1), self.control_bits +  self.data_bits], dtype=np.float32)
        # Set start main control marker.
        inputs[:, 0, 0:self.control_bits] = np.tile(marker_start_main, (self.batch_size, 1))

        # Set bit sequence.
        inputs[:, 1:seq_length+1,  self.control_bits:self.control_bits+self.data_bits] = bit_seq

        for r in range(recall_number):
            # Set start aux serial recall control marker.
            inputs[:, (r+1)*(seq_length+1), 0:self.control_bits] = np.tile(marker_start_aux, (self.batch_size, 1))
            inputs[:, (r+1)*(seq_length+1)+1:(r+2)*(seq_length+1), 0:self.control_bits] = np.tile(ctrl_aux, (self.batch_size, seq_length,1))
        
        # 2. Generate targets.
        # Generate target:  [BATCH_SIZE, 3*SEQ_LENGTH+3, DATA_BITS] (only data bits!)
        targets = np.zeros([self.batch_size, (recall_number+1)*(seq_length+1),  self.data_bits], dtype=np.float32)
        # Set bit sequence for serial recall.
        for r in range(recall_number):
            targets[:, (r+1)*(seq_length+1)+1:(r+2)*(seq_length+1),  :] = np.fliplr(bit_seq)

        # 3. Generate mask.
        # Generate target mask: [BATCH_SIZE, 3*SEQ_LENGTH+3]
        mask = torch.zeros([self.batch_size, (recall_number+1)*(seq_length+1)]).type(torch.ByteTensor)
        for r in range(recall_number):
            mask[:, (r+1)*(seq_length+1)+1:(r+2)*(seq_length+1)] = 1

        # PyTorch variables.
        ptinputs = torch.from_numpy(inputs).type(self.dtype)
        pttargets = torch.from_numpy(targets).type(self.dtype)

        # Return tuples.
        data_tuple = DataTuple(ptinputs, pttargets)
        aux_tuple = AlgSeqAuxTuple(mask, seq_length, recall_number)

        return data_tuple, aux_tuple


    # method for changing the maximum length, used mainly during curriculum learning
    def set_max_length(self, max_length):
        self.max_sequence_length = max_length

if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""
    
    # "Loaded parameters".
    params = {'control_bits': 4, 'data_bits': 8, 'batch_size': 2, 
        #'randomize_control_lines': True,
        'min_sequence_length': 1, 'max_sequence_length': 10,  'bias': 0.5}
    # Create problem object.
    problem = RepeatReverseRecallCommandLines(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    data_tuple, aux_tuple = next(generator)
    # Display single sample (0) from batch.
    problem.show_sample(data_tuple, aux_tuple)
