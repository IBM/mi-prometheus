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

    def __getitem__(self, index):
        """
        Getter that returns one individual sample generated on-the-fly

        .. note::

            The sequence length is drawn randomly between ``self.min_sequence_length`` and \
            ``self.max_sequence_length``.


        :param index: index of the sample to return.

        :return: DataDict({'sequences', 'sequences_length', 'targets', 'mask', 'num_subsequences'}), with:

            - sequences: [2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS],
            - **sequences_length: random value between self.min_sequence_length and self.max_sequence_length**
            - targets: [2*SEQ_LENGTH+2, DATA_BITS],
            - mask: [2*SEQ_LENGTH+2]
            - num_subsequences: 1


        """
        # Set sequence length
        seq_length = np.random.randint(self.min_sequence_length, self.max_sequence_length + 1)

        # Generate batch of random bit sequences [SEQ_LENGTH X DATA_BITS]
        bit_seq = np.random.binomial(1, self.bias, (seq_length, self.data_bits))

        # Generate input:  [2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS]
        inputs = np.zeros([2 * seq_length + 2, self.control_bits + self.data_bits], dtype=np.float32)

        # Set start control marker.
        inputs[0, 0] = 1  # Memorization bit.

        # Set bit sequence.
        inputs[1:seq_length + 1, self.control_bits:self.control_bits + self.data_bits] = bit_seq

        # Set end control marker.
        inputs[seq_length + 1, 1] = 1  # Recall bit.

        # Generate target:  [2*SEQ_LENGTH+2, DATA_BITS] (only data bits!)
        targets = np.zeros([2 * seq_length + 2, self.data_bits], dtype=np.float32)

        # Set bit sequence.
        targets[seq_length + 2:, :] = bit_seq

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
            randomly selected).
            The samples created by ``__getitem__`` are simply not used in this function.


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

        # Set start control marker.
        inputs[:, 0, 0] = 1  # Memorization bit.

        # Set bit sequence.
        inputs[:, 1:seq_length + 1, self.control_bits:self.control_bits + self.data_bits] = bit_seq

        # Set end control marker.
        inputs[:, seq_length + 1, 1] = 1  # Recall bit.

        # Generate target:  [BATCH_SIZE, 2*SEQ_LENGTH+2, DATA_BITS] (only data bits!)
        targets = np.zeros([batch_size, 2 * seq_length + 2,
                            self.data_bits], dtype=np.float32)

        # Set bit sequence.
        targets[:, seq_length + 2:, :] = bit_seq

        # Generate target mask: [BATCH_SIZE, 2*SEQ_LENGTH+2]
        mask = torch.zeros([batch_size, 2 * seq_length + 2]
                           ).type(torch.ByteTensor)
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
    params.add_custom_params({'control_bits': 2,
                              'data_bits': 8,
                              'min_sequence_length': 1,
                              'max_sequence_length': 10})
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
                         shuffle=False, num_workers=4)

    # generate a batch
    import time

    s = time.time()
    for i, batch in enumerate(problem):
        print('Batch # {} - {}'.format(i, type(batch)))

    print('Number of workers: {}'.format(problem.num_workers))
    print('time taken to exhaust a dataset of size {}, with a batch size of {}: {}s'
          .format(serialrecall.__len__(), params['batch_size'], time.time() - s))

    # Display single sample (0) from batch.
    #batch = next(iter(problem))
    #serialrecall.show_sample(batch, 0)
    print('Unit test completed.')
