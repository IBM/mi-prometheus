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
reverse_recall_cl.py: Contains definition of reverse recall problem with control markers and command lines
"""
__author__ = "Ryan McAvoy, Tomasz Kornuta, Vincent Marois"

import torch
import numpy as np
from problems.problem import DataDict
from problems.seq_to_seq.algorithmic.algorithmic_seq_to_seq_problem import AlgorithmicSeqToSeqProblem


class ReverseRecallCommandLines(AlgorithmicSeqToSeqProblem):
    """
    Class generating sequences of random bit-patterns and targets forcing the
    system to learn reverse recall problem.

    1. There are two markers, indicating:
        - beginning of storing/memorization and
        - beginning of recalling from memory.

    2. Additionally, there is a command line (3rd command bit) indicating whether given item is to be\
     stored in memory (0) or recalled (1).


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
        super(ReverseRecallCommandLines, self).__init__(params)

        assert self.control_bits >= 3, "Problem requires at least 3 control bits (currently %r)" % self.control_bits
        assert self.data_bits >= 1, "Problem requires at least 1 data bit (currently %r)" % self.data_bits

        self.name = 'ReverseRecallCommandLines'

        self.randomize_control_lines = params.get(
            'randomize_control_lines', False)

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


        """
        # Define control channel bits.
        # ctrl_main = [0, 0, 0] # not really used.

        # ctrl_aux[2:self.control_bits] = 1 #[0, 0, 1]
        ctrl_aux = np.zeros(self.control_bits)

        if self.control_bits == 3:
            ctrl_aux[2] = 1  # [0, 0, 1]
        else:
            if self.randomize_control_lines:
                # Randomly pick one of the bits to be set.
                ctrl_bit = np.random.randint(2, self.control_bits)
                ctrl_aux[ctrl_bit] = 1
            else:
                ctrl_aux[self.control_bits - 1] = 1

        # Markers.
        marker_start_main = np.zeros(self.control_bits)
        marker_start_main[0] = 1  # [1, 0, 0]
        marker_start_aux = np.zeros(self.control_bits)
        marker_start_aux[1] = 1  # [0, 1, 0]

        # Set sequence length.
        seq_length = np.random.randint(
            self.min_sequence_length, self.max_sequence_length + 1)

        # Generate batch of random bit sequences [SEQ_LENGTH X DATA_BITS]
        bit_seq = np.random.binomial(
            1, self.bias, (seq_length, self.data_bits))

        # 1. Generate inputs.
        # Generate input:  [2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS]
        inputs = np.zeros([2 * seq_length + 2,
                           self.control_bits + self.data_bits],
                          dtype=np.float32)

        # Set start main control marker.
        inputs[0, 0:self.control_bits] = np.tile(
            marker_start_main, (1))

        # Set bit sequence.
        inputs[1:seq_length + 1,
        self.control_bits:self.control_bits + self.data_bits] = bit_seq
        # inputs[:, 1:seq_length+1, 0:self.control_bits] = np.tile(ctrl_main,
        # (self.batch_size, seq_length,1)) # not used as ctrl_main is all
        # zeros.

        # Set start aux control marker.
        inputs[seq_length + 1, 0:self.control_bits] = np.tile(marker_start_aux, (1))
        inputs[seq_length + 2:2 * seq_length + 2, 0:self.control_bits] = np.tile(ctrl_aux, (seq_length, 1))

        # 2. Generate targets.
        # Generate target:  [2*SEQ_LENGTH+2, DATA_BITS] (only data bits!)
        targets = np.zeros([2 * seq_length + 2, self.data_bits], dtype=np.float32)

        # Set bit sequence.
        targets[seq_length + 2:, :] = np.fliplr(bit_seq)

        # 3. Generate mask.
        # Generate target mask: [2*SEQ_LENGTH+2]
        mask = torch.zeros([2 * seq_length + 2]).type(self.app_state.ByteTensor)
        mask[seq_length + 2:] = 1

        # PyTorch variables.
        ptinputs = torch.from_numpy(inputs).type(self.app_state.dtype)
        pttargets = torch.from_numpy(targets).type(self.app_state.dtype)

        # Return data_dict.
        data_dict = DataDict({key: None for key in self.data_definitions.keys()})
        data_dict['sequences'] = ptinputs
        data_dict['sequences_length'] = seq_length
        data_dict['targets'] = pttargets
        data_dict['mask'] = mask
        data_dict['num_subsequences'] = 1

        return data_dict

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

        """
        # get the batch_size
        batch_size = len(batch)

        # Define control channel bits.
        # ctrl_main = [0, 0, 0] # not really used.

        # ctrl_aux[2:self.control_bits] = 1 #[0, 0, 1]
        ctrl_aux = np.zeros(self.control_bits)
        if self.control_bits == 3:
            ctrl_aux[2] = 1  # [0, 0, 1]
        else:
            if self.randomize_control_lines:
                # Randomly pick one of the bits to be set.
                ctrl_bit = np.random.randint(2, self.control_bits)
                ctrl_aux[ctrl_bit] = 1
            else:
                ctrl_aux[self.control_bits - 1] = 1

        # Markers.
        marker_start_main = np.zeros(self.control_bits)
        marker_start_main[0] = 1  # [1, 0, 0]
        marker_start_aux = np.zeros(self.control_bits)
        marker_start_aux[1] = 1  # [0, 1, 0]

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

        # Set start main control marker.
        inputs[:, 0, 0:self.control_bits] = np.tile(
            marker_start_main, (batch_size, 1))

        # Set bit sequence.
        inputs[:, 1:seq_length + 1,
        self.control_bits:self.control_bits + self.data_bits] = bit_seq
        # inputs[:, 1:seq_length+1, 0:self.control_bits] = np.tile(ctrl_main,
        # (self.batch_size, seq_length,1)) # not used as ctrl_main is all
        # zeros.

        # Set start aux control marker.
        inputs[:,
        seq_length + 1,
        0:self.control_bits] = np.tile(marker_start_aux,
                                       (batch_size,
                                        1))
        inputs[:,
        seq_length + 2:2 * seq_length + 2,
        0:self.control_bits] = np.tile(ctrl_aux,
                                       (batch_size,
                                        seq_length,
                                        1))

        # 2. Generate targets.
        # Generate target:  [BATCH_SIZE, 2*SEQ_LENGTH+2, DATA_BITS] (only data
        # bits!)
        targets = np.zeros([batch_size, 2 * seq_length + 2,
                            self.data_bits], dtype=np.float32)
        # Set bit sequence.
        targets[:, seq_length + 2:, :] = np.fliplr(bit_seq)

        # 3. Generate mask.
        # Generate target mask: [BATCH_SIZE, 2*SEQ_LENGTH+2]
        mask = torch.zeros([batch_size, 2 * seq_length + 2]
                           ).type(self.app_state.ByteTensor)
        mask[:, seq_length + 2:] = 1

        # PyTorch variables.
        ptinputs = torch.from_numpy(inputs).type(self.app_state.dtype)
        pttargets = torch.from_numpy(targets).type(self.app_state.dtype)

        # Return data_dict.
        data_dict = DataDict({key: None for key in self.data_definitions.keys()})
        data_dict['sequences'] = ptinputs
        data_dict['sequences_length'] = seq_length
        data_dict['targets'] = pttargets
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
    params.add_config_params({'control_bits': 4,
                              'data_bits': 8,
                              # 'randomize_control_lines': False,
                              'min_sequence_length': 1,
                              'max_sequence_length': 10})
    batch_size = 64

    # Create problem object.
    reverserecallcl = ReverseRecallCommandLines(params)

    # get a sample
    sample = reverserecallcl[0]
    print(repr(sample))
    print('__getitem__ works.')

    # wrap DataLoader on top
    from torch.utils.data.dataloader import DataLoader


    def init_fn(worker_id):
        np.random.seed(seed=worker_id)


    problem = DataLoader(dataset=reverserecallcl, batch_size=batch_size, collate_fn=reverserecallcl.collate_fn,
                         shuffle=False, num_workers=0, worker_init_fn=init_fn)

    # generate a batch
    import time

    s = time.time()
    for i, batch in enumerate(problem):
        print('Batch # {} - {}'.format(i, type(batch)))

    print('Number of workers: {}'.format(problem.num_workers))
    print('time taken to exhaust a dataset of size {}, with a batch size of {}: {}s'
          .format(len(reverserecallcl), batch_size, time.time() - s))

    # Display single sample (0) from batch.
    batch = next(iter(problem))
    reverserecallcl.show_sample(batch, 0)
    print('Unit test completed.')

