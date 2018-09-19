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

__author__ = "Ryan McAvoy & Vincent Marois"

import torch
import numpy as np
from problems.problem import DataDict
from problems.seq_to_seq.algorithmic.algorithmic_seq_to_seq_problem import AlgorithmicSeqToSeqProblem


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
        # Call parent constructor - sets e.g. the loss function, dtype.
        # Additionally it extracts "standard" list of parameters for
        # algorithmic tasks, like batch_size, numbers of bits, sequences etc.
        super(SkipRecallCommandLines, self).__init__(params)

        self.name = 'SkipRecallCommandLines'

        assert self.control_bits >= 3, "Problem requires at least 3 control bits (currently %r)" % self.control_bits
        assert self.data_bits >= 1, "Problem requires at least 1 data bit (currently %r)" % self.data_bits

        self.seq_start = params['seq_start']
        self.skip_length = params['skip_step']

        self.default_values = {'control_bits': self.control_bits,
                               'data_bits': self.data_bits,
                               'min_sequence_length': self.min_sequence_length,
                               'max_sequence_length': self.max_sequence_length,
                               'seq_start': self.seq_start,
                               'skip_length': self.skip_length
                               }

    def __getitem__(self, index):
        """
        Getter that returns one individual sample generated on-the-fly.

        .. note::

            The sequence length is drawn randomly between ``selg.min_sequence_length`` and \
            ``self.max_sequence_length``.


        :param index: index of the sample to return.

        :return: DataDict({'sequences', 'sequences_length', 'targets', 'mask', 'num_subsequences'}), with:

            - sequences: [2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS. Additional elements of sequence are  start and\
                stop control markers, stored in additional bits.

            - **sequences_length: random value between self.min_sequence_length and self.max_sequence_length**
            - targets: [2*SEQ_LENGTH+2, DATA_BITS],
            - mask: [2*SEQ_LENGTH+2]
            - num_subsequences: 1


        # TODO: THE DOCUMENTATION NEEDS TO BE UPDATED
        # TODO: This is commented for now to avoid the issue with `add_ctrl` and `augment` in AlgorithmicSeqToSeqProblem
        # TODO: NOT SURE THAT THIS FN IS WORKING WELL (WITHOUT THE PRESENCE OF THE BATCH DIMENSION)

        """
        '''
        assert (self.max_sequence_length > self.seq_start)

        # define control channel markers
        pos = np.zeros(self.control_bits)  # [0, 0, 0]
        ctrl_data = np.zeros(self.control_bits)  # [0, 0, 0]
        
        ctrl_inter = np.zeros(self.control_bits)
        ctrl_inter[1] = 1  # [0, 1, 0]
        
        ctrl_y = np.zeros(self.control_bits)
        ctrl_y[2] = 1  # [0, 0, 1]
        
        ctrl_dummy = np.ones(self.control_bits)  # [1, 1, 1]
        
        ctrl_start = np.zeros(self.control_bits)
        ctrl_start[0] = 1  # [1, 0, 0]
        
        # assign markers
        markers = ctrl_data, ctrl_dummy, pos

        # Set sequence length
        seq_length = np.random.randint(
            self.min_sequence_length, self.max_sequence_length + 1)

        # Generate batch of random bit sequences [SEQ_LENGTH X DATA_BITS]
        bit_seq = np.random.binomial(1, self.bias, (seq_length, self.data_bits))

        # Generate target by indexing through the array
        target_seq = np.array(bit_seq[self.seq_start::self.skip_length, :])

        #  generate subsequences for x and y
        x = [np.array(bit_seq)]
        
        # data of x and dummies-
        xx = [self.augment(seq, markers, ctrl_start=ctrl_start, add_marker_data=True, add_marker_dummy=False) for seq in x]

        # data of x
        data_1 = [arr for a in xx for arr in a[:-1]]

        # this is a marker between sub sequence x and dummies
        # inter_seq = [add_ctrl(np.zeros((self.batch_size, 1, self.data_bits)), ctrl_inter, pos)]

        # dummies output
        markers2 = ctrl_dummy, ctrl_dummy, pos
        yy = [self.augment(np.zeros(target_seq.shape), markers2, 
                           ctrl_start=ctrl_inter, 
                           add_marker_data=True, 
                           add_marker_dummy=False)]
        data_2 = [arr for a in yy for arr in a[:-1]]

        # add dummies to target
        seq_length_tdummies = seq_length + 2
        dummies_target = np.zeros([seq_length_tdummies, self.data_bits], dtype=np.float32)
        targets = np.concatenate((dummies_target, target_seq), axis=0)

        inputs = np.concatenate(data_1 + data_2, axis=0)

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
        # inputs[:, mask[0], 0:self.control_bits] = ctrl_dummy
        # TODO: fix the batch indexing
        # rest channel values of data dummies

        inputs[:, mask[0], 0:self.control_bits] = torch.tensor(ctrl_y).type(self.app_state.dtype)

        # Return data_dict.
        data_dict = DataDict({key: None for key in self.data_definitions.keys()})
        data_dict['sequences'] = inputs
        data_dict['sequences_length'] = seq_length
        data_dict['targets'] = targets
        data_dict['mask'] = mask
        data_dict['num_subsequences'] = 1
        '''

        return DataDict({key: None for key in self.data_definitions.keys()})  # data_dict

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
            randomly selected). Having several workers does help though, almost cutting the time needed to generate\
            a batch in half according to our experiments.
            The samples created by ``__getitem__`` are simply not used.


        :param batch: Should be a list of DataDict retrieved by `__getitem__`, each containing tensors, numbers,\
        dicts or lists. --> **Not Used Here!**

        :return: DataDict({'sequences', 'sequences_length', 'targets', 'mask', 'num_subsequences'}), with:

            - sequences: [BATCH_SIZE, 2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS],
            - **sequences_length: random value between self.min_sequence_length and self.max_sequence_length**
            - targets: [BATCH_SIZE, 2*SEQ_LENGTH+2, DATA_BITS],
            - mask: [BATCH_SIZE, [2*SEQ_LENGTH+2]
            - num_subsequences: 1

        # TODO: THE DOCUMENTATION NEEDS TO BE UPDATED

        """
        # get the batch_size
        batch_size = len(batch)

        assert (self.max_sequence_length > self.seq_start)

        # define control channel markers

        pos = np.zeros(self.control_bits)  # [0, 0, 0]

        ctrl_data = np.zeros(self.control_bits)  # [0, 0, 0]

        ctrl_inter = np.zeros(self.control_bits)
        ctrl_inter[1] = 1  # [0, 1, 0]

        ctrl_y = np.zeros(self.control_bits)
        ctrl_y[2] = 1  # [0, 0, 1]

        ctrl_dummy = np.ones(self.control_bits)  # [1, 1, 1]

        ctrl_start = np.zeros(self.control_bits)
        ctrl_start[0] = 1  # [1, 0, 0]

        # assign markers
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
        xx = [self.augment(seq, markers, ctrl_start=ctrl_start, add_marker_data=True, add_marker_dummy=False) for seq in x]

        # data of x
        data_1 = [arr for a in xx for arr in a[:-1]]

        # this is a marker between sub sequence x and dummies
        # inter_seq = [add_ctrl(np.zeros((self.batch_size, 1, self.data_bits)), ctrl_inter, pos)]

        # dummies output
        markers2 = ctrl_dummy, ctrl_dummy, pos
        yy = [self.augment(np.zeros(target_seq.shape), markers2, ctrl_start=ctrl_inter, add_marker_data=True,
                           add_marker_dummy=False)]
        data_2 = [arr for a in yy for arr in a[:-1]]

        # add dummies to target
        seq_length_tdummies = seq_length + 2
        dummies_target = np.zeros([batch_size, seq_length_tdummies, self.data_bits], dtype=np.float32)
        targets = np.concatenate((dummies_target, target_seq), axis=1)

        inputs = np.concatenate(data_1 + data_2, axis=1)

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
        # inputs[:, mask[0], 0:self.control_bits] = ctrl_dummy
        # TODO: fix the batch indexing
        # rest channel values of data dummies

        inputs[:, mask[0], 0:self.control_bits] = torch.tensor(ctrl_y).type(self.app_state.dtype)

        # Return data_dict.
        data_dict = DataDict({key: None for key in self.data_definitions.keys()})
        data_dict['sequences'] = inputs
        data_dict['sequences_length'] = seq_length
        data_dict['targets'] = targets
        data_dict['mask'] = mask
        data_dict['num_subsequences'] = 1

        return data_dict

    # method for changing the maximum length, used mainly during curriculum
    # learning
    def set_max_length(self, max_length):
        self.max_sequence_length = max_length


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    from utils.param_interface import ParamInterface

    params = ParamInterface()
    params.add_custom_params({'control_bits': 4,
                              'data_bits': 8,
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
                         shuffle=False, num_workers=4)

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

