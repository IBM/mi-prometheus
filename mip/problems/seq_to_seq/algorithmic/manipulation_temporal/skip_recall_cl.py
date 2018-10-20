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

__author__ = "Ryan McAvoy, Vincent Marois, Tomasz Kornuta"

import torch
import numpy as np
from mip.problems.seq_to_seq.algorithmic.algorithmic_seq_to_seq_problem import AlgorithmicSeqToSeqProblem


class SkipRecallCommandLines(AlgorithmicSeqToSeqProblem):
    """
    Class generating sequences of random bit-patterns and targets forcing the
    system to learn serial recall problem (a.k.a. copy task). The formulation
    follows the original copy task from NTM paper, where:

    1. There are two markers, indicating:
        - beginning of storing/memorization and
        - beginning of recalling from memory.

    2. Additionally, there is a command line (3rd command bit) indicating whether given item is to be stored in
    memory (0) or recalled (1).


    """

    def __init__(self, params):
        """
        Constructor - stores parameters. Calls parent class ``AlgorithmicSeqToSeqProblem``\
         initialization.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).
        """
        params.add_default_params({
            'control_bits': 2,
            'data_bits': 8 })
        # Call parent constructor - sets e.g. the loss function, dtype.
        # Additionally it extracts "standard" list of parameters for
        # algorithmic tasks, like batch_size, numbers of bits, sequences etc.
        super(SkipRecallCommandLines, self).__init__(params)
        self.name = 'SkipRecallCommandLines'

        assert self.control_bits >= 2, "Problem requires at least 2 control bits (currently %r)" % self.control_bits
        assert self.data_bits >= 1, "Problem requires at least 1 data bit (currently %r)" % self.data_bits

        self.seq_start = params['seq_start']
        self.skip_length = params['skip_step']


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
        assert (self.max_sequence_length > self.seq_start)

        # define control channel markers
        ctrl_start_main = np.zeros(self.control_bits)
        ctrl_start_main[self.store_bit] = 1  # [1, 0]

        ctrl_start_aux = np.zeros(self.control_bits)
        ctrl_start_aux[self.recall_bit] = 1  # [0, 1]

        # How cool is the following!:]
        pos = np.zeros(self.control_bits)  # [0, 0]
        ctrl_data = np.zeros(self.control_bits)  # [0, 0]
        ctrl_y = np.zeros(self.control_bits)  # [0, 0]
        ctrl_dummy = np.zeros(self.control_bits)  # [0,0]

        # assign markers - all three [0,0]!
        markers = ctrl_data, ctrl_dummy, pos

        # Set sequence length
        seq_length = np.random.randint(
            self.min_sequence_length, self.max_sequence_length + 1)

        # Generate batch of random bit sequences [BATCH_SIZE x SEQ_LENGTH X DATA_BITS]
        bit_seq = np.random.binomial(
            1, self.bias, (batch_size, seq_length, self.data_bits))

        # Generate target by indexing through the array
        target_seq = np.array(bit_seq[:, self.seq_start::self.skip_length, :])

        #  generate subsequences for x and y
        x = [np.array(bit_seq)]

        # data of x and dummies
        xx = [self.augment(seq, markers, ctrl_start=ctrl_start_main, add_marker_data=True, add_marker_dummy=False) for seq in x]

        # data of x
        data_1 = [arr for a in xx for arr in a[:-1]]

        # this is a marker between sub sequence x and dummies
        # inter_seq = [add_ctrl(np.zeros((self.batch_size, 1, self.data_bits)), ctrl_start_aux, pos)]

        # dummies output - all three [0,0]!!
        markers2 = ctrl_dummy, ctrl_dummy, pos
        yy = [self.augment(np.zeros(target_seq.shape), markers2, ctrl_start=ctrl_start_aux, add_marker_data=True,
                           add_marker_dummy=False)]
        data_2 = [arr for a in yy for arr in a[:-1]]

        # add dummies to target
        seq_length_tdummies = seq_length + 2
        dummies_target = np.zeros([batch_size, seq_length_tdummies, self.data_bits], dtype=np.float32)
        targets = np.concatenate((dummies_target, target_seq), axis=1)

        inputs = np.concatenate(data_1 + data_2, axis=1)
        #inputs[:, , 0:self.control_bits] = ctrl_y

        # 3. Generate mask.
        # Generate target mask: [BATCH_SIZE, 2*SEQ_LENGTH+2, 1]
        ptmasks = torch.zeros([batch_size, inputs.shape[1], 1]
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
    from mip.utils.param_interface import ParamInterface

    params = ParamInterface()
    params.add_config_params({#'control_bits': 4,
                              #'data_bits': 8,
                              'min_sequence_length': 1,
                              'max_sequence_length': 10,
                              'seq_start': 0,
                              'skip_step': 2})
    batch_size = 64

    # Create problem object.
    skiprecallcl = SkipRecallCommandLines(params)

    # get a sample
    sample = skiprecallcl[0]
    print(repr(sample))
    print('__getitem__ works.')

    # wrap DataLoader on top
    from torch.utils.data.dataloader import DataLoader

    problem = DataLoader(dataset=skiprecallcl, batch_size=batch_size, collate_fn=skiprecallcl.collate_fn,
                         shuffle=False, num_workers=0)

    # generate a batch
    import time

    s = time.time()
    for i, batch in enumerate(problem):
        print('Batch # {} - {}'.format(i, type(batch)))

    print('Number of workers: {}'.format(problem.num_workers))
    print('time taken to exhaust a dataset of size {}, with a batch size of {}: {}s'
          .format(skiprecallcl.__len__(), batch_size, time.time() - s))

    # Display single sample (0) from batch.
    batch = next(iter(problem))
    skiprecallcl.show_sample(batch, 0)
    print('Unit test completed.')

