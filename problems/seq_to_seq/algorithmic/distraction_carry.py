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
distraction_carry.py: contains code of distraction carry data generation

"""
__author__ = "Younes Bouhadjar & Vincent Marois"

import torch
import numpy as np
from problems.problem import DataDict
from problems.seq_to_seq.algorithmic.algorithmic_seq_to_seq_problem import AlgorithmicSeqToSeqProblem


class DistractionCarry(AlgorithmicSeqToSeqProblem):
    """
        # TODO: THE DOCUMENTATION OF THIS FILE NEEDS TO BE UPDATED & IMPROVED

    Class generating successions of sub sequences X  and Y of random bit-
    patterns, the target was designed to force the system to learn recalling
    the last sub sequence of Y and all sub sequences of X.
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
        super(DistractionCarry, self).__init__(params)

        self.name = 'DistractionCarry'

        # Assert that numbers of bits are ok.
        assert self.control_bits >= 4, "Problem requires at least 4 control bits (currently %r)" % self.control_bits
        assert self.data_bits >= 1, "Problem requires at least 1 data bit (currently %r)" % self.data_bits

        # Number of subsequences.
        self.num_subseq_min = params["num_subseq_min"]
        self.num_subseq_max = params["num_subseq_max"]

        self.default_values = {'control_bits': self.control_bits,
                               'data_bits': self.data_bits,
                               'min_sequence_length': self.min_sequence_length,
                               'max_sequence_length': self.max_sequence_length,
                               'num_subseq_min': self.num_subseq_min,
                               'num_subseq_max': self.num_subseq_max,
                               }

    def __getitem__(self, index):
        """
        Getter that returns one individual sample generated on-the-fly

        .. note::

            The sequence length is drawn randomly between ``self.min_sequence_length`` and \
            ``self.max_sequence_length``.


        :param index: index of the sample to return.

        :return: DataDict({'sequences', 'sequences_length', 'targets', 'mask', 'num_subsequences'}), with:

            - sequences: [SEQ_LENGTH, CONTROL_BITS+DATA_BITS],
            - **sequences_length: random value between self.min_sequence_length and self.max_sequence_length**
            - targets: [SEQ_LENGTH, DATA_BITS],
            - mask: [SEQ_LENGTH]
            - num_subsequences: 1

            pattern of inputs: # x1 % y1 # x2 % y2 ... # xn % yn & d $ d`
            pattern of target: dummies ...   ...       ...   ...   yn  all(xi)
            mask: used to mask the data part of the target.
            xi, yi, and d(d'): sub sequences x of random length, sub sequence y of random length and dummies.

        # TODO: THE DOCUMENTATION NEEDS TO BE UPDATED
        # TODO: This is commented for now to avoid the issue with `add_ctrl` and `augment` in AlgorithmicSeqToSeqProblem
        # TODO: NOT SURE THAT THIS FN IS WORKING WELL (WITHOUT THE PRESENCE OF THE BATCH DIMENSION)

        """
        '''
        # define control channel markers
        pos = [0, 0, 0, 0]
        ctrl_data = [0, 0, 0, 0]
        ctrl_dummy = [0, 0, 1, 0]
        ctrl_inter = [0, 0, 0, 1]
        # assign markers
        markers = ctrl_data, ctrl_dummy, pos

        # number of sub_sequences
        nb_sub_seq_a = np.random.randint(
            self.num_subseq_min, self.num_subseq_max + 1)
        # might be different in future implementation
        nb_sub_seq_b = nb_sub_seq_a

        # set the sequence length of each marker
        seq_lengths_a = np.random.randint(
            low=self.min_sequence_length,
            high=self.max_sequence_length + 1,
            size=nb_sub_seq_a)
        seq_lengths_b = np.random.randint(
            low=self.min_sequence_length,
            high=self.max_sequence_length + 1,
            size=nb_sub_seq_b)

        #  generate subsequences for x and y
        x = [
            np.random.binomial(
                1,
                self.bias,
                (n,
                 self.data_bits)) for n in seq_lengths_a]
        y = [
            np.random.binomial(
                1,
                self.bias,
                (n,
                 self.data_bits)) for n in seq_lengths_b]

        # create the target
        target_wo_dummies = np.concatenate([y[-1]] + x, axis=0)

        # add marker at the begging of x and dummies
        xx = [
            self.augment(
                seq,
                markers,
                ctrl_start=[
                    1,
                    0,
                    0,
                    0],
                add_marker_data=True,
                add_marker_dummy=False) for seq in x]
        # add marker at the begging of y and dummies of same length, also a
        # marker at the begging of dummies is added
        yy = [self.augment(seq, markers, ctrl_start=[
                           0, 1, 0, 0], add_marker_data=True) for seq in y]

        # this is a marker to separate dummies of x and y at the end of the
        # sequence
        inter_seq = self.add_ctrl(
            np.zeros((1, self.data_bits)), ctrl_inter, pos)

        # data which contains all xs and all ys
        data_1 = [arr for a, b in zip(xx, yy) for arr in a[:-1] + b[:-1]]

        # dummies of y and xs
        data_2 = [yy[-1][-1]] + [inter_seq] + [a[-1] for a in xx]

        # concatenate all parts of the inputs
        inputs = np.concatenate(data_1 + data_2, axis=0)

        # PyTorch variables
        inputs = torch.from_numpy(inputs).type(self.app_state.dtype)
        target_wo_dummies = torch.from_numpy(
            target_wo_dummies).type(self.app_state.dtype)

        # create the mask
        mask_all = inputs[:, 0:self.control_bits] == 1
        mask = mask_all[..., 0]
        for i in range(self.control_bits):
            mask = mask_all[..., i] * mask

        # rest ctrl channel of dummies
        inputs[mask[0], 0:self.control_bits] = 0

        # Create the target with the dummies
        targets = torch.zeros_like(inputs[:, self.control_bits:])
        targets[mask[0], :] = target_wo_dummies

        # Return data_dict.
        data_dict = DataDict({key: None for key in self.data_definitions.keys()})
        data_dict['sequences'] = inputs
        data_dict['sequences_length'] = max(seq_lengths_a)
        data_dict['targets'] = targets
        data_dict['mask'] = mask
        data_dict['num_subsequences'] = nb_sub_seq_a + nb_sub_seq_b
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

       pattern of inputs: # x1 % y1 # x2 % y2 ... # xn % yn & d $ d`
       pattern of target: dummies ...   ...       ...   ...   yn  all(xi)
       mask: used to mask the data part of the target.
       xi, yi, and d(d'): sub sequences x of random length, sub sequence y of random length and dummies.


        # TODO: THE DOCUMENTATION NEEDS TO BE UPDATED & IMPROVED

        """
        # get the batch_size
        batch_size = len(batch)

        # define control channel markers
        pos = [0, 0, 0, 0]
        ctrl_data = [0, 0, 0, 0]
        ctrl_dummy = [0, 0, 1, 0]
        ctrl_inter = [0, 0, 0, 1]
        # assign markers
        markers = ctrl_data, ctrl_dummy, pos

        # number of sub_sequences
        nb_sub_seq_a = np.random.randint(
            self.num_subseq_min, self.num_subseq_max + 1)
        # might be different in future implementation
        nb_sub_seq_b = nb_sub_seq_a

        # set the sequence length of each marker
        seq_lengths_a = np.random.randint(
            low=self.min_sequence_length,
            high=self.max_sequence_length + 1,
            size=nb_sub_seq_a)
        seq_lengths_b = np.random.randint(
            low=self.min_sequence_length,
            high=self.max_sequence_length + 1,
            size=nb_sub_seq_b)

        #  generate subsequences for x and y
        x = [
            np.random.binomial(
                1,
                self.bias,
                (batch_size,
                 n,
                 self.data_bits)) for n in seq_lengths_a]
        y = [
            np.random.binomial(
                1,
                self.bias,
                (batch_size,
                 n,
                 self.data_bits)) for n in seq_lengths_b]

        # create the target
        target_wo_dummies = np.concatenate([y[-1]] + x, axis=1)

        # add marker at the begging of x and dummies
        xx = [
            self.augment(
                seq,
                markers,
                ctrl_start=[
                    1,
                    0,
                    0,
                    0],
                add_marker_data=True,
                add_marker_dummy=False) for seq in x]
        # add marker at the begging of y and dummies of same length, also a
        # marker at the begging of dummies is added
        yy = [self.augment(seq, markers, ctrl_start=[
            0, 1, 0, 0], add_marker_data=True) for seq in y]

        # this is a marker to separate dummies of x and y at the end of the
        # sequence
        inter_seq = self.add_ctrl(
            np.zeros((batch_size, 1, self.data_bits)), ctrl_inter, pos)

        # data which contains all xs and all ys
        data_1 = [arr for a, b in zip(xx, yy) for arr in a[:-1] + b[:-1]]

        # dummies of y and xs
        data_2 = [yy[-1][-1]] + [inter_seq] + [a[-1] for a in xx]

        # concatenate all parts of the inputs
        inputs = np.concatenate(data_1 + data_2, axis=1)

        # PyTorch variables
        inputs = torch.from_numpy(inputs).type(self.app_state.dtype)
        target_wo_dummies = torch.from_numpy(
            target_wo_dummies).type(self.app_state.dtype)

        # create the mask
        mask_all = inputs[:, :, 0:self.control_bits] == 1
        mask = mask_all[..., 0]
        for i in range(self.control_bits):
            mask = mask_all[..., i] * mask

        # rest ctrl channel of dummies
        inputs[:, mask[0], 0:self.control_bits] = 0

        # Create the target with the dummies
        targets = torch.zeros_like(inputs[:, :, self.control_bits:])
        targets[:, mask[0], :] = target_wo_dummies

        # Return data_dict.
        data_dict = DataDict({key: None for key in self.data_definitions.keys()})
        data_dict['sequences'] = inputs
        data_dict['sequences_length'] = max(seq_lengths_a)
        data_dict['targets'] = targets
        data_dict['mask'] = mask
        data_dict['num_subsequences'] = nb_sub_seq_a + nb_sub_seq_b

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
                              'num_subseq_min': 1,
                              'num_subseq_max': 4})
    batch_size = 64

    # Create problem object.
    distractioncarry = DistractionCarry(params)

    # get a sample
    sample = distractioncarry[0]
    print(repr(sample))
    print('__getitem__ works.')

    # wrap DataLoader on top
    from torch.utils.data.dataloader import DataLoader

    problem = DataLoader(dataset=distractioncarry, batch_size=batch_size, collate_fn=distractioncarry.collate_fn,
                         shuffle=False, num_workers=4)

    # generate a batch
    import time

    s = time.time()
    for i, batch in enumerate(problem):
        print('Batch # {} - {}'.format(i, type(batch)))

    print('Number of workers: {}'.format(problem.num_workers))
    print('time taken to exhaust a dataset of size {}, with a batch size of {}: {}s'
          .format(distractioncarry.__len__(), batch_size, time.time() - s))

    # Display single sample (0) from batch.
    batch = next(iter(problem))
    distractioncarry.show_sample(batch, 0)
    print('Unit test completed.')
