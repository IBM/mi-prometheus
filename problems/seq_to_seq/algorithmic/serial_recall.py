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
serial_recall_original.py: Original serial recall problem (a.k.a. copy task)

"""
__author__ = "Tomasz Kornuta, Younes Bouhadjar, Vincent Marois"

import torch
import numpy as np

from types import MethodType

from problems.problem import DataDict
from problems.seq_to_seq.algorithmic.algorithmic_seq_to_seq_problem import AlgorithmicSeqToSeqProblem


class SerialRecall(AlgorithmicSeqToSeqProblem):
    """
    Class generating sequences of random bit-patterns and targets forcing the
    system to learn serial recall problem (a.k.a. copy task). The formulation
    follows the original copy task from NTM paper, where:

    1. There are two markers, indicating:

        - beginning of storing/memorization and
        - beginning of recalling from memory.

    2. For other elements of the sequence the command bits are set to zero

    3. Minor modification I: the target contains only data bits (command bits are skipped)

    4. Minor modification II: generator returns a mask, which can be used for filtering important elements of the output.


    TODO: sequences of different lengths in batch (filling with zeros?)

    """

    def __init__(self, params):
        """
        Constructor - stores parameters. Calls parent class ``AlgorithmicSeqToSeqProblem``\
         initialization.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).
        """
        # Call parent constructor - sets e.g. the loss function, dtype.
        # Additionally it extracts "standard" list of parameters for
        # algorithmic tasks, like batch_size, numbers of bits, sequences etc.
        super(SerialRecall, self).__init__(params)

        self.name = 'SerialRecall'

        assert self.control_bits >= 2, "Problem requires at least 2 control bits (currently %r)" % self.control_bits
        assert self.data_bits >= 2, "Problem requires at least 1 data bit (currently %r)" % self.data_bits


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
        # Set sequence length
        seq_length = np.random.randint(self.min_sequence_length,
                                       self.max_sequence_length + 1)

        # Generate batch of random bit sequences [BATCH_SIZE x SEQ_LENGTH X DATA_BITS]
        bit_seq = np.random.binomial(1, self.bias,
                                     (batch_size, seq_length, self.data_bits))

        # Generate input:  [BATCH_SIZE, 2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS]
        inputs = np.zeros([batch_size, 2 * seq_length + 2,
                           self.control_bits + self.data_bits],
                          dtype=np.float32)

        # Set control marker with store bit.
        inputs[:, 0, self.store_bit] = 1 

        # Set bit sequence.
        inputs[:, 1:seq_length + 1, self.control_bits:self.control_bits + self.data_bits] = bit_seq

        # Set control marker with recall bit.
        inputs[:, seq_length + 1, self.recall_bit] = 1  

        # Generate target:  [BATCH_SIZE, 2*SEQ_LENGTH+2, DATA_BITS] (only data bits!)
        targets = np.zeros([batch_size, 2 * seq_length + 2,
                            self.data_bits], dtype=np.float32)

        # Set bit sequence.
        targets[:, seq_length + 2:, :] = bit_seq

        # Generate target mask: [BATCH_SIZE, 2*SEQ_LENGTH+2, 1]
        ptmasks = torch.zeros([batch_size, 2 * seq_length + 2, 1]
                           ).type(torch.ByteTensor)
        ptmasks[:, seq_length + 2:, 0] = 1
        
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
    params.add_config_params({'control_bits': 2,
                              'data_bits': 8,
                              'min_sequence_length': 1,
                              'max_sequence_length': 10}),
                              #'generation_mode': 'optimized'})
    batch_size = 64

    # Create problem object.
    serialrecall = SerialRecall(params)

    # get a sample
    sample = serialrecall[0]
    print(repr(sample))
    print('__getitem__ works.')

    # wrap DataLoader on top
    from torch.utils.data.dataloader import DataLoader
    problem = DataLoader(dataset=serialrecall, batch_size=batch_size, collate_fn=serialrecall.collate_fn,
                         shuffle=False, num_workers=0, worker_init_fn=serialrecall.worker_init_fn)

    # generate a batch
    import time

    s = time.time()
    for i, batch in enumerate(problem):
        #print('Batch # {} - {}'.format(i, type(batch)))
        pass

    print('Number of workers: {}'.format(problem.num_workers))
    print('time taken to exhaust a dataset of size {}, with a batch size of {}: {}s'
          .format(serialrecall.__len__(), batch_size, time.time() - s))

    # Display single sample (0) from batch.
    batch = next(iter(problem))
    serialrecall.show_sample(batch, 0)
    print('Unit test completed.')

