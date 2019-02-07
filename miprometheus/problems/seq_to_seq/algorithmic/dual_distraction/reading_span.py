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

"""reading_span.py: contains code of reading span data generation"""
__author__ = "Younes Bouhadjar, Vincent Marois"

import torch
import numpy as np
from miprometheus.utils.data_dict import DataDict
from miprometheus.problems.seq_to_seq.algorithmic.algorithmic_seq_to_seq_problem import AlgorithmicSeqToSeqProblem


class ReadingSpan(AlgorithmicSeqToSeqProblem):
    """
    # TODO : Documentation will be added soon
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
        super(ReadingSpan, self).__init__(params)

        self.name = 'ReadingSpan'

        assert self.control_bits >= 2, "Problem requires at least 2 control bits (currently %r)" % self.control_bits
        assert self.data_bits >= 1, "Problem requires at least 1 data bit (currently %r)" % self.data_bits

        # Number of subsequences.
        self.num_subseq_min = params["num_subseq_min"]
        self.num_subseq_max = params["num_subseq_max"]

    def __getitem__(self, index):
        """
        Getter that returns one individual sample generated on-the-fly

        .. note::

            The sequence length is drawn randomly between ``self.min_sequence_length`` and \
            ``self.max_sequence_length``.


        :param index: index of the sample to return.

        :return: DataDict({'sequences', 'sequences_length', 'targets', 'mask', 'num_subsequences'}), with:

            - sequences: [SEQ_LENGTH, CONTROL_BITS+DATA_BITS]. SEQ_LENGTH depends on number of sub-sequences
                and its lengths.

            - **sequences_length: random value between self.min_sequence_length and self.max_sequence_length**
            - targets: [SEQ_LENGTH, DATA_BITS],
            - mask: [SEQ_LENGTH]
            - num_subsequences: number of subsequences


        """
        # TODO: THE DOCUMENTATION NEEDS TO BE UPDATED
        # TODO: This is commented for now to avoid the issue with `add_ctrl` and `augment` in AlgorithmicSeqToSeqProblem
        # TODO: NOT SURE THAT THIS FN IS WORKING WELL (WITHOUT THE PRESENCE OF THE BATCH DIMENSION)
        '''
        # define control channel markers
        pos = [0, 0]
        ctrl_data = [0, 0]
        ctrl_dummy = [0, 1]
        ctrl_inter = [0, 1]

        # assign markers
        markers = ctrl_data, ctrl_dummy, pos

        # number sub sequences
        num_sub_seq = np.random.randint( self.num_subseq_min, self.num_subseq_max + 1)

        # set the sequence length of each marker
        seq_length = np.random.randint(low=self.min_sequence_length, high=self.max_sequence_length + 1, 
                                       size=num_sub_seq)

        #  generate subsequences for x and y
        x = [np.random.binomial(1, self.bias, (n, self.data_bits)) for n in seq_length]
        x_last = [a[None, -1, :] for a in x]

        # create the target
        seq_length_tdummies = sum(seq_length) + seq_length.shape[0] + 1
        
        dummies_target = np.zeros([seq_length_tdummies, self.data_bits], dtype=np.float32)
        targets = np.concatenate([dummies_target] + x_last, axis=0)

        # data of x and dummies
        xx = [self.augment(seq, markers, ctrl_start=[1, 0], add_marker_data=True, add_marker_dummy=False) for seq in x]

        # data of x
        data_1 = [arr for a in xx for arr in a[:-1]]

        # this is a marker between sub sequence x and dummies
        inter_seq = self.add_ctrl(np.zeros((1, self.data_bits)), ctrl_inter, pos)

        # dummies of x
        x_dummy_last = [a[None, -1, :] for b in xx for a in b[-1:]]

        # concatenate all parts of the inputs
        inputs = np.concatenate(data_1 + [inter_seq] + x_dummy_last, axis=0)

        # PyTorch variables
        inputs = torch.from_numpy(inputs).type(self.app_state.dtype)
        targets = torch.from_numpy(targets).type(self.app_state.dtype)
        
        # TODO: batch might have different sequence lengths
        mask_all = inputs[..., 0:self.control_bits] == 1
        mask = mask_all[..., 0]
        for i in range(self.control_bits):
            mask = mask_all[..., i] * mask

        # TODO: fix the batch indexing
        # rest channel values of data dummies
        inputs[mask[0], 0:self.control_bits] = 0

        # Return data_dict.
        data_dict = DataDict({key: None for key in self.data_definitions.keys()})
        data_dict['sequences'] = inputs
        data_dict['sequences_length'] = max(seq_length)
        data_dict['targets'] = targets
        data_dict['mask'] = mask
        data_dict['num_subsequences'] = num_sub_seq
        '''

        return DataDict({key: None for key in self.data_definitions.keys()}) #data_dict

    def collate_fn(self, batch):
        """
        Generates a batch of samples on-the-fly

        .. warning::
            Because of the fact that the sequence length is randomly drawn between ``self.min_sequence_length`` and \
            ``self.max_sequence_length`` and then fixed for one given batch (**but varies between batches**), \
            we cannot follow the scheme `merge together individuals samples that can be retrieved in parallel with\
            several workers.` Indeed, each sample could have a different sequence length, and merging them together\
            would then not be possible (we cannot have variable-sequence-length samples within one batch \
            without padding).
            Hence, ``collate_fn`` generates on-the-fly a batch of samples, all having the same length (initially\
            randomly selected).
            The samples created by ``__getitem__`` are simply not used in this function.


        :param batch: Should be a list of DataDict retrieved by `__getitem__`, each containing tensors, numbers,\
        dicts or lists. --> **Not Used Here!**

        :return: DataDict({'sequences', 'sequences_length', 'targets', 'mask', 'num_subsequences'}), with:

            - sequences: [BATCH_SIZE, SEQ_LENGTH, CONTROL_BITS+DATA_BITS],
            - **sequences_length: random value between self.min_sequence_length and self.max_sequence_length**
            - targets: [BATCH_SIZE, SEQ_LENGTH, DATA_BITS],
            - mask: [BATCH_SIZE, SEQ_LENGTH]
            - num_subsequences: number of subsequences

        # TODO: THE DOCUMENTATION NEEDS TO BE UPDATED
        """

        # get the batch_size
        batch_size = len(batch)

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
        seq_length = np.random.randint(low=self.min_sequence_length, high=self.max_sequence_length + 1,
                                       size=num_sub_seq)

        #  generate subsequences for x and y
        x = [np.random.binomial(1, self.bias, (batch_size, n, self.data_bits)) for n in seq_length]
        x_last = [a[:, None, -1, :] for a in x]

        # create the target
        seq_length_tdummies = sum(seq_length) + seq_length.shape[0] + 1
        dummies_target = np.zeros([batch_size, seq_length_tdummies, self.data_bits], dtype=np.float32)
        targets = np.concatenate([dummies_target] + x_last, axis=1)

        # data of x and dummies
        xx = [self.augment(seq, markers, ctrl_start=[1, 0], add_marker_data=True, add_marker_dummy=False) for seq in x]

        # data of x
        data_1 = [arr for a in xx for arr in a[:-1]]

        # this is a marker between sub sequence x and dummies
        inter_seq = self.add_ctrl(np.zeros((batch_size, 1, self.data_bits)), ctrl_inter, pos)

        # dummies of x
        x_dummy_last = [a[:, None, -1, :] for b in xx for a in b[-1:]]

        # concatenate all parts of the inputs
        inputs = np.concatenate(data_1 + [inter_seq] + x_dummy_last, axis=1)

        # PyTorch variables
        inputs = torch.from_numpy(inputs).type(self.app_state.dtype)
        targets = torch.from_numpy(targets).type(self.app_state.dtype)

        # TODO: batch might have different sequence lengths
        mask_all = inputs[..., 0:self.control_bits] == 1
        mask = mask_all[..., 0]
        for i in range(self.control_bits):
            mask = mask_all[..., i] * mask

        # TODO: fix the batch indexing
        # rest channel values of data dummies
        inputs[:, mask[0], 0:self.control_bits] = 0

        # Return data_dict.
        data_dict = DataDict({key: None for key in self.data_definitions.keys()})
        data_dict['sequences'] = inputs
        data_dict['sequences_length'] = max(seq_length)
        data_dict['targets'] = targets
        data_dict['mask'] = mask
        data_dict['num_subsequences'] = num_sub_seq

        return data_dict

    # method for changing the maximum length, used mainly during curriculum
    # learning
    def set_max_length(self, max_length):
        self.max_sequence_length = max_length


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    from miprometheus.utils.param_interface import ParamInterface

    params = ParamInterface()
    params.add_config_params({'control_bits': 2,
                              'data_bits': 8,
                              'batch_size': 2,
                              'min_sequence_length': 1,
                              'max_sequence_length': 10,
                              'num_subseq_min': 4,
                              'num_subseq_max': 4})
    batch_size = 64

    # Create problem object.
    readingspan = ReadingSpan(params)

    # get a sample
    sample = readingspan[0]
    print(repr(sample))
    print('__getitem__ works.')

    # wrap DataLoader on top
    from torch.utils.data import DataLoader

    problem = DataLoader(dataset=readingspan, batch_size=batch_size, collate_fn=readingspan.collate_fn,
                         shuffle=False, num_workers=0)

    # generate a batch
    import time

    s = time.time()
    for i, batch in enumerate(problem):
        print('Batch # {} - {}'.format(i, type(batch)))

    print('Number of workers: {}'.format(problem.num_workers))
    print('time taken to exhaust a dataset of size {}, with a batch size of {}: {}s'
          .format(readingspan.__len__(), batch_size, time.time() - s))

    # Display single sample (0) from batch.
    batch = next(iter(problem))
    readingspan.show_sample(batch, 0)
    print('Unit test completed.')

