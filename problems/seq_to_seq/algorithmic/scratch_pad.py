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

"""scratch_pad.py: contains code of scratch recall data generation
"""
__author__ = "Younes Bouhadjar & Vincent Marois"

import torch
import numpy as np
from problems.problem import DataDict
from problems.seq_to_seq.algorithmic.algorithmic_seq_to_seq_problem import AlgorithmicSeqToSeqProblem


class ScratchPad(AlgorithmicSeqToSeqProblem):
    """
    Class generating sequences of random bit-patterns and targets forcing the
    system to learn the scratch pad problem (overwriting the memory).
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
        super(ScratchPad, self).__init__(params)

        self.name = 'ScratchPad'

        assert self.control_bits >= 2, "Problem requires at least 2 control bits (currently %r)" % self.control_bits
        assert self.data_bits >= 1, "Problem requires at least 1 data bit (currently %r)" % self.data_bits

        # Number of subsequences.
        self.num_subseq_min = params["num_subseq_min"]
        self.num_subseq_max = params["num_subseq_max"]

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

            - sequences: [BATCH_SIZE, SEQ_LENGTH, CONTROL_BITS+DATA_BITS],
            - sequences_length: [BATCH_SIZE] (random value between self.min_sequence_length and self.max_sequence_length)
            - targets: [BATCH_SIZE, SEQ_LENGTH, DATA_BITS],
            - masks: [BATCH_SIZE, SEQ_LENGTH, 1]
            - num_subsequences: [BATCH_SIZE, 1] (number of subsequences)

        """

        # define control channel markers
        pos = [0, 0]
        ctrl_data = [0, 0]
        ctrl_dummy = [0, 1]
        ctrl_inter = [0, 1]

        # assign markers
        markers = ctrl_data, ctrl_dummy, pos

        # number sub sequences
        num_sub_seq = np.random.randint(self.num_subseq_min, self.num_subseq_max + 1)

        # set the sequence length of each marker
        seq_lengths = np.random.randint(low=self.min_sequence_length, high=self.max_sequence_length + 1,
                                       size=num_sub_seq)

        #  generate subsequences for x and y
        x = [np.random.binomial(1, self.bias, (batch_size, n, self.data_bits)) for n in seq_lengths]

        # create the target
        seq_length_tdummies = sum(seq_lengths) + seq_lengths.shape[0] + 1
        dummies_target = np.zeros([batch_size, seq_length_tdummies, self.data_bits], dtype=np.float32)
        targets = np.concatenate((dummies_target, x[-1]), axis=1)

        # data of x and dummies
        xx = [self.augment(seq, markers, ctrl_start=[1, 0],
                           add_marker_data=True,
                           add_marker_dummy=False) for seq in x]

        # data of x
        data_1 = [arr for a in xx for arr in a[:-1]]

        # this is a marker between sub sequence x and dummies
        inter_seq = self.add_ctrl(np.zeros((batch_size, 1, self.data_bits)), ctrl_inter, pos)

        # dummies of x
        data_2 = [xx[-1][-1]]

        # concatenate all parts of the inputs
        inputs = np.concatenate(data_1 + [inter_seq] + data_2, axis=1)

        # Generate 3D ByteTensor for mask.
        ptmasks = torch.zeros([batch_size, inputs.shape[1], 1]).type(torch.ByteTensor)
        ptmasks[:, inputs.shape[1]-seq_lengths[-1]:, 0] = 1

        # Return data_dict.
        data_dict = self.create_data_dict()
        data_dict['sequences'] = torch.from_numpy(inputs).type(self.app_state.dtype)
        data_dict['targets'] = torch.from_numpy(targets).type(self.app_state.dtype)
        data_dict['masks'] = ptmasks
        data_dict['sequences_length'] = torch.ones([batch_size, 1]).type(torch.CharTensor) * max(seq_lengths).item()
        data_dict['num_subsequences'] = torch.ones([batch_size, 1]).type(torch.CharTensor) * num_sub_seq
        return data_dict


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    from utils.param_interface import ParamInterface

    params = ParamInterface()
    params.add_config_params({'control_bits': 2,
                              'data_bits': 8,
                              'min_sequence_length': 1,
                              'max_sequence_length': 10,
                              'num_subseq_min': 2,
                              'num_subseq_max': 4})
    batch_size = 10

    # Create problem object.
    scratchpad = ScratchPad(params)

    # get a sample
    sample = scratchpad[0]
    print(repr(sample))
    print('__getitem__ works.')

    # wrap DataLoader on top
    from torch.utils.data.dataloader import DataLoader

    problem = DataLoader(dataset=scratchpad, batch_size=batch_size, collate_fn=scratchpad.collate_fn,
                         shuffle=False, num_workers=0)

    # generate a batch
    import time

    s = time.time()
    for i, batch in enumerate(problem):
        #print('Batch # {} - {}'.format(i, type(batch)))
        pass

    print('Number of workers: {}'.format(problem.num_workers))
    print('time taken to exhaust a dataset of size {}, with a batch size of {}: {}s'
          .format(scratchpad.__len__(), batch_size, time.time() - s))

    # Display single sample (0) from batch.
    batch = next(iter(problem))
    scratchpad.show_sample(batch, 0)
    print('Unit test completed.')

