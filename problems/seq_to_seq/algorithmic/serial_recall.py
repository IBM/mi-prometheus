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

        # "Attach" the "__getitem__" and "collate_fn" functions to class.
        #setattr(self.__class__, '__getitem__', staticmethod(self.generate_sample_ignore_index))
        #setattr(self.__class__, 'collate_fn', staticmethod(self.collate_samples_from_batch))
        setattr(self.__class__, '__getitem__', staticmethod(self.generate_sample_ignore_index))
        setattr(self.__class__, 'collate_fn', staticmethod(self.collate_by_batch_generation))


    def pad_collate_tensor_list(self, tensor_list, max_seq_len = -1):
        """
            Method collates list of 2D tensors with varying dimension 0 ("sequence length").
            Pads 0 along that dimension.

            :param tensor_list: list [BATCH_SIZE] of tensors [SEQ_LEN, DATA_SIZE] to be padded.
            :param max_seq_len: max sequence length (DEFAULT: -1 means that it will recalculate it on the fly)

            :return: 3D padded tensor [BATCH_SIZE, MAX_SEQ_LEN, DATA_SIZE]

        """
        # Get batch size.
        batch_size = len(tensor_list)

        if (max_seq_len < 0):
            # Get max total length.
            max_seq_len = max([t.shape[0] for t in tensor_list])

        # Collate tensors - add padding to each of them separatelly.
        collated_tensors = torch.zeros(size=(batch_size, max_seq_len, tensor_list[0].shape[-1]))
        for i,t in enumerate(tensor_list):
            # Version 1: pad
            #ten_pad = max_seq_len - t.shape[0]
            # (padLeft, padRight, padTop, padBottom)
            #pad = torch.nn.ZeroPad2d( (0, 0, 0, ten_pad))
            #collated_tensors[i,:,:] = pad(t)
            # Version 2: copy.
            ten_len = t.shape[0]
            collated_tensors[i,:ten_len] = t

        return collated_tensors


    def generate_batch(self, batch_size):
        """
        Generates a batch of samples on-the-fly.

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

        # Generate target mask: [BATCH_SIZE, 2*SEQ_LENGTH+2, 1]
        ptmasks = torch.zeros([batch_size, 2 * seq_length + 2, 1]
                           ).type(torch.ByteTensor)
        ptmasks[:, seq_length + 2:, 0] = 1

        # PyTorch variables.
        ptinputs = torch.from_numpy(inputs).type(torch.DoubleTensor)
        pttargets = torch.from_numpy(targets).type(torch.DoubleTensor)

        # Set seqnence length - all have the same length.
        ptseq_length = torch.ones([batch_size,1]).type(torch.CharTensor) * seq_length
        

        # Return data_dict.
        data_dict = DataDict({key: None for key in self.data_definitions.keys()})
        data_dict['sequences'] = ptinputs
        data_dict['sequences_length'] = ptseq_length
        data_dict['targets'] = pttargets
        data_dict['masks'] = ptmasks
        data_dict['num_subsequences'] = torch.ones([batch_size, 1]).type(torch.CharTensor)

        return data_dict


    def generate_sample_ignore_index(self, index):
        """
        Returns one individual sample generated on-the-fly.

        .. note::

            The sequence length is drawn randomly between ``self.min_sequence_length`` and \
            ``self.max_sequence_length``.

        .. warning::

            As the name of the method suggests, ''the index'' will in fact be ignored during generation.

        :param index: index of the sample to returned (IGNORED).

        :return: DataDict({'sequences', 'sequences_length', 'targets', 'masks', 'num_subsequences'}), with:

            - sequences: [2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS],
            - sequences_length: [1] (random value between self.min_sequence_length and self.max_sequence_length)
            - targets: [2*SEQ_LENGTH+2, DATA_BITS]
            - masks: [2*SEQ_LENGTH+2]
            - num_subsequences: [1]

        """
        # Generate batch of size 1.
        data_dict = self.generate_batch(1)

        # Squeeze the batch dimension.
        for key in self.data_definitions.keys():
            data_dict[key] = data_dict[key].squeeze(0)

        return data_dict



    def collate_samples_from_batch(self, batch_of_dicts):
        """
        Generates a batch of samples on-the-fly

        :param batch_of_dicts: Should be a list of DataDict retrieved by `__getitem__`, each containing tensors, numbers,\
        dicts or lists. --> **Not Used Here!**

        :return: DataDict({'sequences', 'sequences_length', 'targets', 'masks', 'num_subsequences'}), with:

            - sequences: [BATCH_SIZE, 2*MAX_SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS],
            - sequences_length: [BATCH_SIZE, 1] (random values between self.min_sequence_length and self.max_sequence_length)
            - targets: [BATCH_SIZE, 2*MAX_SEQ_LENGTH+2, DATA_BITS],
            - mask: [BATCH_SIZE, [2*MAX_SEQ_LENGTH+2]
            - num_subsequences: [BATCH_SIZE, 1]

        """
        # Get max total (input+markers+output) length.
        max_batch_total_len = max([d['sequences'].shape[0] for d in batch_of_dicts])

        # Collate sequences - add padding to each of them separatelly.
        collated_sequences = self.pad_collate_tensor_list(
            [d['sequences'] for d in batch_of_dicts], max_batch_total_len)
        print(collated_sequences.shape)

        # Collate masks.
        collated_masks = self.pad_collate_tensor_list(
            [d['masks'] for d in batch_of_dicts], max_batch_total_len)
        print(collated_masks.shape)

        # Collate targets.
        collated_targets = self.pad_collate_tensor_list(
            [d['targets'] for d in batch_of_dicts], max_batch_total_len)
        print(collated_targets.shape)

        # Collate lengths.
        collated_lengths = torch.tensor([d['sequences_length'] for d in batch_of_dicts])
        print(collated_lengths)

        # Collate lengths.
        collated_num_subsequences = torch.tensor([d['num_subsequences'] for d in batch_of_dicts])
        print(collated_num_subsequences)

        # Return data_dict.
        data_dict = DataDict({key: None for key in self.data_definitions.keys()})
        data_dict['sequences'] = collated_sequences
        data_dict['sequences_length'] = collated_lengths
        data_dict['targets'] = collated_targets
        data_dict['masks'] = collated_masks
        data_dict['num_subsequences'] = collated_num_subsequences

        return data_dict        


    def do_not_generate_sample(self, index):
        """
        Method used as __getitem__ in "optimized" mode.
        It simply returns back the received index.
        Whole generation is made in  ''collate_fn'' (i.e. collate_by_generation_batch'')

        .. warning::

            As the name of the method suggests, the method does not generate the sample.

        :param index: index of the sample to returned (IGNORED).

        :return: index

        """
        return index



    def collate_by_batch_generation(self, batch):
        """
        Generates a batch of samples on-the-fly.

        .. warning::
            The samples created by ``__getitem__`` are simply not used in this function.
            As``collate_fn`` generates on-the-fly a batch of samples relying on the underlying ''generate_batch''\
            method, all having the same length (randomly selected thought).

        :param batch: **Not Used Here!**

        :return: DataDict({'sequences', 'sequences_length', 'targets', 'masks', 'num_subsequences'}), with:

            - sequences: [BATCH_SIZE, 2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS]
            - sequences_length: [BATCH_SIZE, 1] (the same random value between self.min_sequence_length and self.max_sequence_length)
            - targets: [BATCH_SIZE, , 2*SEQ_LENGTH+2, DATA_BITS]
            - masks: [BATCH_SIZE, 2*SEQ_LENGTH+2, 1]
            - num_subsequences: [BATCH_SIZE, 1]

        """
        # Generate batch of size 1.
        data_dict = self.generate_batch(len(batch))

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
    params.add_config_params({'control_bits': 2,
                              'data_bits': 2,
                              'min_sequence_length': 1,
                              'max_sequence_length': 10})
    batch_size = 10

    # Create problem object.
    serialrecall = SerialRecall(params)

    # get a sample
    sample = serialrecall[0]
    #print(repr(sample))
    #print('__getitem__ works.')

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
    #batch = next(iter(problem))
    #serialrecall.show_sample(batch, 0)
    print('Unit test completed.')

