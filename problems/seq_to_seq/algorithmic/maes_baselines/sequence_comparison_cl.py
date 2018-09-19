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


class SequenceComparisonCommandLines(AlgorithmicSeqToSeqProblem):
    """
    # TODO: THE DOCUMENTATION NEEDS TO BE UPDATED

    Class generating sequences of random bit-patterns and targets forcing the
    system to learn scratch pad problem (overwrite the memory).

    @Ryan: ARE YOU SURE? FIX THE CLASS DESCRIPTION!

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
        super(SequenceComparisonCommandLines, self).__init__(params)

        assert self.control_bits >= 3, "Problem requires at least 3 control bits (currently %r)" % self.control_bits
        assert self.data_bits >= 1, "Problem requires at least 1 data bit (currently %r)" % self.data_bits

        self.name = 'SequenceComparisonCommandLines'

        # The bit that indicates whether we want to return true when items are
        # equal or not equal
        self.predict_inverse = params.get('predict_inverse', True)

    def __getitem__(self, index):
        """
        Getter that returns one individual sample generated on-the-fly.

        .. note::

            The sequence length is drawn randomly between ``selg.min_sequence_length`` and \
            ``self.max_sequence_length``.


        :param index: index of the sample to return.

        :return: DataDict({'sequences', 'sequences_length', 'targets', 'mask', 'num_subsequences'}), with:

            - sequences: [SEQ_LENGTH, CONTROL_BITS+DATA_BITS. Additional elements of sequence are  start and\
                stop control markers, stored in additional bits.

            - **sequences_length: random value between self.min_sequence_length and self.max_sequence_length**
            - targets: [SEQ_LENGTH, DATA_BITS],
            - mask: [SEQ_LENGTH]
            - num_subsequences: 1

          pattern of inputs: x1, x2, ...xn d
          pattern of target: d, d,   ...d xn
          mask: used to mask the data part of the target
          xi, d: sub sequences, dummies

        # TODO: THE DOCUMENTATION NEEDS TO BE UPDATED
        # TODO: This is commented for now to avoid the issue with `add_ctrl` and `augment` in AlgorithmicSeqToSeqProblem
        # TODO: NOT SURE THAT THIS FN IS WORKING WELL (WITHOUT THE PRESENCE OF THE BATCH DIMENSION)

        """
        '''
        # define control channel markers
        # pos = [0, 0, 0]
        pos = np.zeros(self.control_bits)  # [0, 0, 0]
        # ctrl_data = [0, 0, 0]
        ctrl_data = np.zeros(self.control_bits)  # [0, 0, 0]

        # ctrl_inter = [0, 1, 0]
        ctrl_inter = np.zeros(self.control_bits)
        ctrl_inter[1] = 1  # [0, 1, 0]

        # ctrl_output = [1, 1, 1]
        ctrl_output = np.ones(self.control_bits)  # [1, 1, 1]

        # ctrl_dummy = [0, 0, 1]
        ctrl_dummy = np.zeros(self.control_bits)
        ctrl_dummy[2] = 1  # [0, 0, 1]

        # ctrl_start = [1, 0, 0]
        ctrl_start = np.zeros(self.control_bits)
        ctrl_start[0] = 1  # [1, 0, 0]
        # assign markers
        markers = ctrl_data, ctrl_dummy, pos

        # set the sequence length of each marker
        seq_length = np.random.randint(
            low=self.min_sequence_length, high=self.max_sequence_length + 1)

        #  generate subsequences for x and y
        x = [np.array(np.random.binomial(
            1, self.bias, (seq_length, self.data_bits)))]

        # Generate the second sequence which is either a scrambled version of the first
        # or exactly identical with approximately 50% probability (technically the scrambling
        # allows them to be the same with a very low chance)

        # First generate a random binomial of the same size as x, this will be
        # used be used with an xor operation to scamble x to get y
        xor_scrambler = np.array(np.random.binomial(1, self.bias, x[0].shape))

        # Create a mask that will set entire batches of the xor_scrambler to zero. The batches that are zero
        # will force the xor to return the original x for that batch
        scrambler_mask = np.array(np.random.binomial(
            1, self.bias, (seq_length)))
        xor_scrambler = np.array(
            xor_scrambler * scrambler_mask[:, :, np.newaxis])

        aux_seq = np.array(np.logical_xor(x[0], xor_scrambler))

        if self.predict_inverse:
            # if the xor scambler is all zeros then x and y will be the same so
            # target will be true
            actual_target = np.array(
                np.any(xor_scrambler, axis=2, keepdims=True))
        else:
            actual_target = np.logical_not(
                np.array(np.any(xor_scrambler, axis=2, keepdims=True)))

        # create the target
        seq_length_tdummies = seq_length + 2
        dummies_target = np.zeros(
            [seq_length_tdummies, 1], dtype=np.float32)
        target = np.concatenate((dummies_target, actual_target), axis=1)

        # data of x and dummies
        xx = [
            self.augment(
                seq,
                markers,
                ctrl_start=ctrl_start,
                add_marker_data=True,
                add_marker_dummy=False) for seq in x]

        # data of x
        data_1 = [arr for a in xx for arr in a[:-1]]

        # this is a marker between sub sequence x and dummies
        inter_seq = [self.add_ctrl(
            np.zeros((1, self.data_bits)), ctrl_inter, pos)]
        # Second Sequence for comparison

        markers2 = ctrl_output, ctrl_dummy, pos
        yy = [self.augment(aux_seq, markers2, ctrl_start=ctrl_output,
                           add_marker_data=False, add_marker_dummy=False)]
        data_2 = [arr for a in yy for arr in a[:-1]]

        recall_seq = [self.add_ctrl(
            np.zeros((1, self.data_bits)), ctrl_dummy, pos)]
        dummy_data = [
            self.add_ctrl(
                np.zeros(
                    (1, self.data_bits)), np.ones(
                    len(ctrl_dummy)), pos)]

        # concatenate all parts of the inputs
        inputs = np.concatenate(data_1 + inter_seq + data_2, axis=1)

        # PyTorch variables
        inputs = torch.from_numpy(inputs).type(self.app_state.dtype)
        target = torch.from_numpy(target).type(self.app_state.dtype)
        # TODO: batch might have different sequence lengths
        mask_all = inputs[..., 0:self.control_bits] == 1
        mask = mask_all[..., 0]
        for i in range(self.control_bits):
            mask = mask_all[..., i] * mask

        # rest channel values of data dummies
        inputs[:, mask[0], 0:self.control_bits] = torch.tensor(ctrl_dummy).type(self.app_state.dtype)

        # Return data_dict.
        data_dict = DataDict({key: None for key in self.data_definitions.keys()})
        data_dict['sequences'] = inputs
        data_dict['sequences_length'] = seq_length
        data_dict['targets'] = target
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

          pattern of inputs: x1, x2, ...xn d
          pattern of target: d, d,   ...d xn
          mask: used to mask the data part of the target
          xi, d: sub sequences, dummies

        # TODO: THE DOCUMENTATION NEEDS TO BE UPDATED

        """
        # get the batch_size
        batch_size = len(batch)

        # define control channel markers
        # pos = [0, 0, 0]
        pos = np.zeros(self.control_bits)  # [0, 0, 0]
        # ctrl_data = [0, 0, 0]
        ctrl_data = np.zeros(self.control_bits)  # [0, 0, 0]

        # ctrl_inter = [0, 1, 0]
        ctrl_inter = np.zeros(self.control_bits)
        ctrl_inter[1] = 1  # [0, 1, 0]

        # ctrl_output = [1, 1, 1]
        ctrl_output = np.ones(self.control_bits)  # [1, 1, 1]

        # ctrl_dummy = [0, 0, 1]
        ctrl_dummy = np.zeros(self.control_bits)
        ctrl_dummy[2] = 1  # [0, 0, 1]

        # ctrl_start = [1, 0, 0]
        ctrl_start = np.zeros(self.control_bits)
        ctrl_start[0] = 1  # [1, 0, 0]
        # assign markers
        markers = ctrl_data, ctrl_dummy, pos

        # set the sequence length of each marker
        seq_length = np.random.randint(
            low=self.min_sequence_length, high=self.max_sequence_length + 1)

        #  generate subsequences for x and y
        x = [np.array(np.random.binomial(
            1, self.bias, (batch_size, seq_length, self.data_bits)))]

        # Generate the second sequence which is either a scrambled version of the first
        # or exactly identical with approximately 50% probability (technically the scrambling
        # allows them to be the same with a very low chance)

        # First generate a random binomial of the same size as x, this will be
        # used be used with an xor operation to scamble x to get y
        xor_scrambler = np.array(np.random.binomial(1, self.bias, x[0].shape))

        # Create a mask that will set entire batches of the xor_scrambler to zero. The batches that are zero
        # will force the xor to return the original x for that batch
        scrambler_mask = np.array(np.random.binomial(
            1, self.bias, (batch_size, seq_length)))
        xor_scrambler = np.array(
            xor_scrambler * scrambler_mask[:, :, np.newaxis])

        aux_seq = np.array(np.logical_xor(x[0], xor_scrambler))

        if self.predict_inverse:
            # if the xor scambler is all zeros then x and y will be the same so
            # target will be true
            actual_target = np.array(
                np.any(xor_scrambler, axis=2, keepdims=True))
        else:
            actual_target = np.logical_not(
                np.array(np.any(xor_scrambler, axis=2, keepdims=True)))

        # create the target
        seq_length_tdummies = seq_length + 2
        dummies_target = np.zeros(
            [batch_size, seq_length_tdummies, 1], dtype=np.float32)
        target = np.concatenate((dummies_target, actual_target), axis=1)

        # data of x and dummies
        xx = [
            self.augment(
                seq,
                markers,
                ctrl_start=ctrl_start,
                add_marker_data=True,
                add_marker_dummy=False) for seq in x]

        # data of x
        data_1 = [arr for a in xx for arr in a[:-1]]

        # this is a marker between sub sequence x and dummies
        inter_seq = [self.add_ctrl(
            np.zeros((batch_size, 1, self.data_bits)), ctrl_inter, pos)]
        # Second Sequence for comparison

        markers2 = ctrl_output, ctrl_dummy, pos
        yy = [self.augment(aux_seq, markers2, ctrl_start=ctrl_output,
                           add_marker_data=False, add_marker_dummy=False)]
        data_2 = [arr for a in yy for arr in a[:-1]]

        recall_seq = [self.add_ctrl(
            np.zeros((batch_size, 1, self.data_bits)), ctrl_dummy, pos)]
        dummy_data = [
            self.add_ctrl(
                np.zeros(
                    (batch_size, 1, self.data_bits)), np.ones(
                    len(ctrl_dummy)), pos)]

        # concatenate all parts of the inputs
        inputs = np.concatenate(data_1 + inter_seq + data_2, axis=1)

        # PyTorch variables
        inputs = torch.from_numpy(inputs).type(self.app_state.dtype)
        target = torch.from_numpy(target).type(self.app_state.dtype)
        # TODO: batch might have different sequence lengths
        mask_all = inputs[..., 0:self.control_bits] == 1
        mask = mask_all[..., 0]
        for i in range(self.control_bits):
            mask = mask_all[..., i] * mask

        # rest channel values of data dummies
        inputs[:, mask[0], 0:self.control_bits] = torch.tensor(
            ctrl_dummy).type(self.app_state.dtype)

        # Return data_dict.
        data_dict = DataDict({key: None for key in self.data_definitions.keys()})
        data_dict['sequences'] = inputs
        data_dict['sequences_length'] = seq_length
        data_dict['targets'] = target
        data_dict['mask'] = mask
        data_dict['num_subsequences'] = 1

        return data_dict

    def generate_batch(self):
        """
        Generates a batch  of size [BATCH_SIZE, SEQ_LENGTH,
        CONTROL_BITS+DATA_BITS]. SEQ_LENGTH depends on number of sub-sequences
        and its lengths.

        :returns: Tuple consisting of: input, output and mask
                  pattern of inputs: x1, x2, ...xn d
                  pattern of target: d, d,   ...d xn
                  mask: used to mask the data part of the target
                  xi, d: sub sequences, dummies

        """
        # define control channel markers
        # pos = [0, 0, 0]
        pos = np.zeros(self.control_bits)  # [0, 0, 0]
        # ctrl_data = [0, 0, 0]
        ctrl_data = np.zeros(self.control_bits)  # [0, 0, 0]

        # ctrl_inter = [0, 1, 0]
        ctrl_inter = np.zeros(self.control_bits)
        ctrl_inter[1] = 1  # [0, 1, 0]

        # ctrl_output = [1, 1, 1]
        ctrl_output = np.ones(self.control_bits)  # [1, 1, 1]

        # ctrl_dummy = [0, 0, 1]
        ctrl_dummy = np.zeros(self.control_bits)
        ctrl_dummy[2] = 1  # [0, 0, 1]

        #ctrl_start = [1, 0, 0]
        ctrl_start = np.zeros(self.control_bits)
        ctrl_start[0] = 1  # [1, 0, 0]
        # assign markers
        markers = ctrl_data, ctrl_dummy, pos

        # set the sequence length of each marker
        seq_length = np.random.randint(
            low=self.min_sequence_length, high=self.max_sequence_length + 1)

        #  generate subsequences for x and y
        x = [np.array(np.random.binomial(
            1, self.bias, (self.batch_size, seq_length, self.data_bits)))]

        # Generate the second sequence which is either a scrambled version of the first
        # or exactly identical with approximately 50% probability (technically the scrambling
        # allows them to be the same with a very low chance)

        # First generate a random binomial of the same size as x, this will be
        # used be used with an xor operation to scamble x to get y
        xor_scrambler = np.array(np.random.binomial(1, self.bias, x[0].shape))

        # Create a mask that will set entire batches of the xor_scrambler to zero. The batches that are zero
        # will force the xor to return the original x for that batch
        scrambler_mask = np.array(np.random.binomial(
            1, self.bias, (self.batch_size, seq_length)))
        xor_scrambler = np.array(
            xor_scrambler * scrambler_mask[:, :, np.newaxis])

        aux_seq = np.array(np.logical_xor(x[0], xor_scrambler))

        if self.predict_inverse:
            # if the xor scambler is all zeros then x and y will be the same so
            # target will be true
            actual_target = np.array(
                np.any(xor_scrambler, axis=2, keepdims=True))
        else:
            actual_target = np.logical_not(
                np.array(np.any(xor_scrambler, axis=2, keepdims=True)))

        # create the target
        seq_length_tdummies = seq_length + 2
        dummies_target = np.zeros(
            [self.batch_size, seq_length_tdummies, 1], dtype=np.float32)
        target = np.concatenate((dummies_target, actual_target), axis=1)

        # data of x and dummies
        xx = [
            self.augment(
                seq,
                markers,
                ctrl_start=ctrl_start,
                add_marker_data=True,
                add_marker_dummy=False) for seq in x]

        # data of x
        data_1 = [arr for a in xx for arr in a[:-1]]

        # this is a marker between sub sequence x and dummies
        inter_seq = [self.add_ctrl(
            np.zeros((self.batch_size, 1, self.data_bits)), ctrl_inter, pos)]
        # Second Sequence for comparison

        markers2 = ctrl_output, ctrl_dummy, pos
        yy = [self.augment(aux_seq, markers2, ctrl_start=ctrl_output,
                           add_marker_data=False, add_marker_dummy=False)]
        data_2 = [arr for a in yy for arr in a[:-1]]

        recall_seq = [self.add_ctrl(
            np.zeros((self.batch_size, 1, self.data_bits)), ctrl_dummy, pos)]
        dummy_data = [
            self.add_ctrl(
                np.zeros(
                    (self.batch_size, 1, self.data_bits)), np.ones(
                    len(ctrl_dummy)), pos)]

        # concatenate all parts of the inputs
        inputs = np.concatenate(data_1 + inter_seq + data_2, axis=1)

        # PyTorch variables
        inputs = torch.from_numpy(inputs).type(self.dtype)
        target = torch.from_numpy(target).type(self.dtype)
        # TODO: batch might have different sequence lengths
        mask_all = inputs[..., 0:self.control_bits] == 1
        mask = mask_all[..., 0]
        for i in range(self.control_bits):
            mask = mask_all[..., i] * mask

        # rest channel values of data dummies
        inputs[:, mask[0], 0:self.control_bits] = torch.tensor(
            ctrl_dummy).type(self.dtype)

        # Return data tuple.
        data_tuple = DataTuple(inputs, target)
        # Returning maximum length of sequence a - for now.
        aux_tuple = AlgSeqAuxTuple(mask, seq_length, 1)

        return data_tuple, aux_tuple

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
                              # 'predict_inverse': False,
                              'min_sequence_length': 1,
                              'max_sequence_length': 3})
    batch_size = 64

    # Create problem object.
    seqcompcl = SequenceComparisonCommandLines(params)

    # get a sample
    sample = seqcompcl[0]
    print(repr(sample))
    print('__getitem__ works.')

    # wrap DataLoader on top
    from torch.utils.data.dataloader import DataLoader


    def init_fn(worker_id):
        np.random.seed(seed=worker_id)


    problem = DataLoader(dataset=seqcompcl, batch_size=batch_size, collate_fn=seqcompcl.collate_fn,
                         shuffle=False, num_workers=4, worker_init_fn=init_fn)

    # generate a batch
    import time

    s = time.time()
    for i, batch in enumerate(problem):
        print('Batch # {} - {}'.format(i, type(batch)))

    print('Number of workers: {}'.format(problem.num_workers))
    print('time taken to exhaust a dataset of size {}, with a batch size of {}: {}s'
          .format(len(seqcompcl), batch_size, time.time() - s))

    # Display single sample (0) from batch.
    batch = next(iter(problem))
    seqcompcl.show_sample(batch, 0)
    print('Unit test completed.')
