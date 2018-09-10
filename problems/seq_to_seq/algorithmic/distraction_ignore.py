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

"""distraction_ignore.py: contains code of distraction ignore data generation"""
__author__ = "Younes Bouhadjar"

import torch
import numpy as np
from problems.problem import DataTuple
from problems.seq_to_seq.algorithmic.algorithmic_seq_to_seq_problem import AlgorithmicSeqToSeqProblem, AlgSeqAuxTuple
from misc.param_interface import ParamInterface


class DistractionIgnore(AlgorithmicSeqToSeqProblem):
    """
    Class generating successions of sub sequences X  and Y of random bit-
    patterns, the target was designed to force the system to learn recalling
    just sub sequences X and ignore Y.
    """

    def __init__(self, params):
        """
        Constructor - stores parameters. Calls parent class initialization.

        :param params: Dictionary of parameters.
        """
        # Call parent constructor - sets e.g. the loss function, dtype.
        # Additionally it extracts "standard" list of parameters for
        # algorithmic tasks, like batch_size, numbers of bits, sequences etc.
        super(DistractionIgnore, self).__init__(params)

        assert self.control_bits >= 4, "Problem requires at least 4 control bits (currently %r)" % self.control_bits
        assert self.data_bits >= 1, "Problem requires at least 1 data bit (currently %r)" % self.data_bits

        # Number of subsequences.
        self.num_subseq_min = params["num_subseq_min"]
        self.num_subseq_max = params["num_subseq_max"]

    def generate_batch(self):
        """
        Generates a batch  of size [BATCH_SIZE, SEQ_LENGTH,
        CONTROL_BITS+DATA_BITS]. SEQ_LENGTH depends on number of sub-sequences
        and its lengths.

        :returns: Tuple consisting of: input, output and mask
                  pattern of inputs: # x1 % y1 # x2 % y2 ... # xn % yn & d
                  pattern of target: dummies ...   ...       ...   ...   all(xi)
                  mask: used to mask the data part of the target.
                  xi, yi, and d: sub sequences x of random length, sub sequence y of random length and dummies.

        """
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
                (self.batch_size,
                 n,
                 self.data_bits)) for n in seq_lengths_a]
        y = [
            np.random.binomial(
                1,
                self.bias,
                (self.batch_size,
                 n,
                 self.data_bits)) for n in seq_lengths_b]

        # create the target
        target = np.concatenate(x, axis=1)

        # add marker at the begging of x and dummies of same length
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

        # add marker at the begging of y and dummies of same length,  also a marker at the begging of dummies is added
        # TODO: as we don't need the dummies here (no y needs recalling), we
        # should add an arguements specifying if dummies are needed or not
        yy = [self.augment(seq, markers, ctrl_start=[
                           0, 1, 0, 0], add_marker_data=True) for seq in y]

        # this is a marker to separate dummies of x and y at the end of the
        # sequence
        inter_seq = self.add_ctrl(
            np.zeros((self.batch_size, 1, self.data_bits)), ctrl_inter, pos)

        # data which contains all xs and all ys
        data_1 = [arr for a, b in zip(xx, yy) for arr in a[:-1] + b[:-1]]

        # dummies of y and xs
        data_2 = [inter_seq] + [a[-1] for a in xx]

        # concatenate all parts of the inputs
        inputs = np.concatenate(data_1 + data_2, axis=1)

        # PyTorch variables
        inputs = torch.from_numpy(inputs).type(self.dtype)
        target = torch.from_numpy(target).type(self.dtype)

        # create the mask
        mask_all = inputs[:, :, 0:self.control_bits] == 1
        mask = mask_all[..., 0]
        for i in range(self.control_bits):
            mask = mask_all[..., i] * mask

        # rest ctrl channel of dummies
        inputs[:, mask[0], 0:self.control_bits] = 0

        # Create the target with the dummies
        target_with_dummies = torch.zeros_like(
            inputs[:, :, self.control_bits:])
        target_with_dummies[:, mask[0], :] = target

        # Return data tuple.
        data_tuple = DataTuple(inputs, target_with_dummies)
        # Returning maximum length of sequence a - for now.
        aux_tuple = AlgSeqAuxTuple(
            mask, max(seq_lengths_a), nb_sub_seq_a + nb_sub_seq_b)

        return data_tuple, aux_tuple

    # method for changing the maximum length, used mainly during curriculum
    # learning
    def set_max_length(self, max_length):
        self.max_sequence_length = max_length


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    params = ParamInterface()
    params.add_custom_params({'control_bits': 4,
                              'data_bits': 8,
                              'batch_size': 1,
                              'min_sequence_length': 1,
                              'max_sequence_length': 10,
                              'num_subseq_min': 1,
                              'num_subseq_max': 4})
    # Create problem object.
    problem = DistractionIgnore(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    data_tuple, aux_tuple = next(generator)
    # Display single sample (0) from batch.
    problem.show_sample(data_tuple, aux_tuple)
