#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
sequence_symmetry_cl.py: Contains implementation of sequence symmetry task.

"""
__author__ = "Tomasz Kornuta, Ryan McAvoy, Vincent Marois"

import torch
import numpy as np
from problems.problem import DataDict
from problems.seq_to_seq.algorithmic.algorithmic_seq_to_seq_problem import AlgorithmicSeqToSeqProblem


class SequenceSymmetryCommandLines(AlgorithmicSeqToSeqProblem):
    """
        Class generating sequences of random bit-patterns and targets forcing the
        system to learn sequence symmetry task. Two sequences x1 and x2 are
        symmetric if x2 == reversed(x1).

        ..note:
            Can also work in ''antisymmetry'' mode, i.e. return 1 when x1 != reversed(x2).

    """

    def __init__(self, params):
        """
        Constructor - stores parameters. Calls parent class ``AlgorithmicSeqToSeqProblem``\
         initialization.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).
        """
        # Set default number of bits for a given problem.
        # This has to be done before calling base class constructor!
        params.add_default_params({
            'control_bits': 2,
            'data_bits': 8 })
        # Call parent constructor - sets e.g. the loss function, dtype.
        # Additionally it extracts "standard" list of parameters for
        # algorithmic tasks, like batch_size, numbers of bits, sequences etc.
        super(SequenceSymmetryCommandLines, self).__init__(params)
        self.name = 'SequenceSymmetryCommandLines'

        # Overwrite default value of output item size to 1!
        self.default_values['output_item_size'] = 1

        assert self.control_bits >= 2, "Problem requires at least 2 control bits (currently %r)" % self.control_bits
        assert self.data_bits >= 1, "Problem requires at least 1 data bit (currently %r)" % self.data_bits

        # The bit that indicates whether we want to return true when sequences
        # are symmetric or not.
        self.params.add_default_params({'antisymmetry': False})
        self.antisymmetry = params['antisymmetry']

        # Level "Hard": scrambles only single sample instead of the whole batch.  
        self.params.add_default_params({'hard': False})
        self.hard = params['hard']


    def generate_batch(self, batch_size):
        """
        Generates a batch of samples of size ''batch_size'' on-the-fly.

       .. note::

            The sequence length is drawn randomly between ``self.min_sequence_length`` and \
            ``self.max_sequence_length``.

       .. warning::
            All the samples within the batch will have the same sequence lengt.

        :param batch_size: Size of the batch to be returned. 

        :return: DataDict({'sequences', 'sequences_length', 'targets', 'masks', 'num_subsequences'}), with:

            - sequences: [BATCH_SIZE, 2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS]
            - sequences_length: [BATCH_SIZE, 1] (the same random value between self.min_sequence_length and self.max_sequence_length)
            - targets: [BATCH_SIZE, , 2*SEQ_LENGTH+2, DATA_BITS]
            - masks: [BATCH_SIZE, 2*SEQ_LENGTH+2, 1]
            - num_subsequences: [BATCH_SIZE, 1]

        """
        # Store marker.
        marker_start_main = np.zeros(self.control_bits)
        marker_start_main[self.store_bit] = 1  # [1, 0, 0]

        # Recall marker.
        marker_start_aux = np.zeros(self.control_bits)
        marker_start_aux[self.recall_bit] = 1  # [0, 1, 0]

        # Define control lines.
        ctrl_aux = np.zeros(self.control_bits)
        if self.use_control_lines:
            if  self.control_bits >= 3:
                if self.randomize_control_lines:
                    # Randomly pick one of the bits to be set.
                    ctrl_bit = np.random.randint(2, self.control_bits)
                    ctrl_aux[ctrl_bit] = 1
                else:
                    # Set last.
                    ctrl_aux[self.control_bits - 1] = 1
        # Else: no control lines!

        # Set sequence length.
        seq_length = np.random.randint(
            self.min_sequence_length, self.max_sequence_length + 1)

        # Generate batch of random bit sequences [BATCH_SIZE x SEQ_LENGTH X
        # DATA_BITS]
        bit_seq = np.random.binomial(
            1, self.bias, (batch_size, seq_length, self.data_bits))

        # 1. Generate inputs.
        # Generate input:  [BATCH_SIZE, 2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS]
        inputs = np.zeros([batch_size,
                           2 * seq_length + 2,
                           self.control_bits + self.data_bits],
                          dtype=np.float32)

        # Set store control marker.
        inputs[:, 0, 0:self.control_bits] = np.tile(
            marker_start_main, (batch_size, 1))

        # Set input items.
        inputs[:, 1:seq_length + 1,
            self.control_bits:self.control_bits + self.data_bits] = bit_seq

        # Set recall control marker.
        inputs[:,seq_length + 1, 0:self.control_bits] = np.tile(
            marker_start_aux, (batch_size, 1))

        # Set control lines for recall items.   
        inputs[:,seq_length + 2:2 * seq_length + 2,0:self.control_bits] = np.tile(
            ctrl_aux,(batch_size,seq_length,1))

        # Check if second subsequence has to be symmetrical.
        batch_symmetrical = np.random.random_sample(batch_size) < 0.5
        #print("batch_symmetrical =\n",batch_symmetrical)

        # Generate scambler mask.
        scrambler_mask = np.random.binomial(1, self.bias,
            (batch_size, seq_length, self.data_bits))
        #print(scrambler_mask)

        # Create the second bit sequence.                
        aux_bit_seq = np.copy(np.fliplr(bit_seq))
        # Iterate through samples (sequences) in batch.
        for i, symmetrical in enumerate(batch_symmetrical):
            if not symmetrical:
                if self.hard:
                    # Pick one item from sequence.
                    item_number = np.random.random_integers(0, seq_length-1) 
                    # Scramble it.
                    aux_bit_seq[i, item_number, : ] = np.logical_xor(
                        aux_bit_seq[i, item_number, : ], scrambler_mask[i, item_number, : ])
                else:
                    # Scramble the whole sequence.
                    aux_bit_seq[i, :, : ] = np.logical_xor(
                        aux_bit_seq[i, :, : ], scrambler_mask[i, :, : ])
        #print(aux_bit_seq)

        # Set bit sequence.
        inputs[:, seq_length + 2:2 * seq_length + 2, 
            self.control_bits:self.control_bits + self.data_bits] = aux_bit_seq

        # 2. Generate targets.
        # Generate target:  [BATCH_SIZE, 2*SEQ_LENGTH+2, 1] (only 1 bit!)
        targets = np.zeros([batch_size, 2 * seq_length + 2, 1], dtype=np.float32)

        # Check once again if all items/sequences are equal - just in case.
        are_items_different = np.sum(aux_bit_seq != np.fliplr(bit_seq), axis=2) > 0
        batch_symmetrical = np.sum(are_items_different, axis=1) == 0
        #print("batch_symmetrical =\n",batch_symmetrical)

        # Set only last output item.
        # Check symmetry/antisimmetry mode.
        if self.antisymmetry:
             batch_symmetrical = np.logical_not(batch_symmetrical)
        targets[:, -1, 0] = batch_symmetrical

        # Generate target mask: [BATCH_SIZE, 2*SEQ_LENGTH+2, 1]
        ptmasks = torch.zeros([batch_size, 2 * seq_length + 2, 1]
                           ).type(self.app_state.ByteTensor)
        ptmasks[:, -1] = 1

        # Return data_dict.
        data_dict = self.create_data_dict()
        data_dict['sequences'] = torch.from_numpy(inputs).type(self.app_state.dtype)
        data_dict['targets'] = torch.from_numpy(targets).type(self.app_state.dtype)
        data_dict['masks'] = ptmasks
        data_dict['sequences_length'] = torch.ones([batch_size,1]).type(torch.CharTensor) * seq_length
        data_dict['num_subsequences'] = torch.ones([batch_size, 1]).type(torch.CharTensor)
        return data_dict


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    from utils.param_interface import ParamInterface

    params = ParamInterface()
    params.add_config_params({#'control_bits': 2,
                              #'data_bits': 8,
                              #'antisymmetry': True,
                              'hard' : True,
                              'min_sequence_length': 3,
                              'max_sequence_length': 5})
    batch_size = 64

    # Create problem object.
    seqsymcl = SequenceSymmetryCommandLines(params)

    # get a sample
    sample = seqsymcl[0]
    print(repr(sample))
    print('__getitem__ works.')

    # wrap DataLoader on top
    from torch.utils.data.dataloader import DataLoader

    problem = DataLoader(dataset=seqsymcl, batch_size=batch_size, collate_fn=seqsymcl.collate_fn,
                         shuffle=False, num_workers=0)

    # generate a batch
    import time

    s = time.time()
    for i, batch in enumerate(problem):
        #print('Batch # {} - {}'.format(i, type(batch)))
        pass

    print('Number of workers: {}'.format(problem.num_workers))
    print('time taken to exhaust a dataset of size {}, with a batch size of {}: {}s'
          .format(seqsymcl.__len__(), batch_size, time.time() - s))

    # Display single sample (0) from batch.
    batch = next(iter(problem))
    seqsymcl.show_sample(batch, 0)
    print('Unit test completed.')


