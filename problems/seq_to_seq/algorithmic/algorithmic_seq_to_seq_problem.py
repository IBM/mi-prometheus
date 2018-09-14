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

"""algorithmic_seq_to_seq_problem.py: abstract base class for algorithmic, sequential problems"""
__author__ = "Tomasz Kornuta, Younes Bouhadjar"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from problems.problem import DataDict
from problems.seq_to_seq.seq_to_seq_problem import SeqToSeqProblem
from utils.loss.masked_bce_with_logits_loss import MaskedBCEWithLogitsLoss


class AlgorithmicSeqToSeqProblem(SeqToSeqProblem):
    """
    Base class for algorithmic, sequential problems.

    Provides some basic functionality usefull in all problems of such
    type

    """

    def __init__(self, params):
        """
        Initializes problem object. Calls base constructor. Sets
        nn.BCEWithLogitsLoss() as default loss function.

        :param params: Dictionary of parameters (read from configuration file).

        """
        super(AlgorithmicSeqToSeqProblem, self).__init__(params)

        # Set default loss function - cross entropy.
        if self.use_mask:
            self.loss_function = MaskedBCEWithLogitsLoss()
        else:
            self.loss_function = nn.BCEWithLogitsLoss()

        # Extract "standard" list of parameters for algorithmic tasks.

        # Number of bits in one element.
        self.control_bits = params['control_bits']
        self.data_bits = params['data_bits']

        # Min and max lengths of a single subsequence (number of elements).
        self.min_sequence_length = params['min_sequence_length']
        self.max_sequence_length = params['max_sequence_length']

        # Add parameter denoting 0-1 distribution (DEFAULT: 0.5 i.e. equal).
        if 'bias' not in params:
            params.add_default_params({'bias': 0.5})
        self.bias = params['bias']

        # Set initial dtype.
        self.dtype = torch.FloatTensor

        # "Default" problem name.
        self.name = 'AlgorithmicSeqToSeqProblem'

        # set default data_definitions dict
        self.data_definitions = {'sequences': {'size': [-1, -1, -1], 'type': [torch.Tensor]},
                                 'sequences_length': {'size': [-1, 1], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1, -1, -1], 'type': [torch.Tensor]},
                                 'mask': {'size': [-1, -1], 'type': [torch.Tensor]},
                                 'num_subsequences': {'size': [-1, 1], 'type': [torch.Tensor]}, #TODO: check size & type
                                 }

        self.default_values = {'control_bits': self.control_bits,
                               'data_bits': self.data_bits,
                               'min_sequence_length': self.min_sequence_length,
                               'max_sequence_length': self.max_sequence_length
                               }

        # This a safety net in case the user forgets to set it
        # because the dataset can be gigantic!
        self.length = 100000

    def calculate_accuracy(self, data_dict, logits):
        """ Calculate accuracy equal to mean difference between outputs and targets.
        WARNING: Applies mask to both logits and targets!

        :param data_dict: DataDict({'sequences', 'sequences_length', 'targets', 'mask', 'num_subsequences'}).

        :param logits: Predictions being output of the model.
        """

        # Check if mask should be is used - if so, apply.
        if self.use_mask:
            return self.loss_function.masked_accuracy(
                logits, data_dict['targets'], data_dict['mask'])
        else:
            return (1 - torch.abs(torch.round(F.sigmoid(logits)) -
                                  data_dict['targets'])).mean()

    # def set_max_length(self, max_length):
    #    """ Sets maximum sequence lenth (property).
    #
    #    :param max_length: Length to be saved as max.
    #    """
    #    self.max_sequence_length = max_length

    def __getitem__(self, index):
        """
        Getter that returns an individual sample from the problem's associated dataset (that can be generated \
        on-the-fly, or retrieved from disk. It can also possibly be composed of several files.).

        To be redefined in subclasses.

        **The getter should return a DataDict: its keys should be defined by `self.data_definitions` keys.**

        e.g.:
            >>> data_dict = DataDict({key: None for key in self.data_definitions.keys()})
            >>> # you can now access each value by its key and assign the corresponding object (e.g. `torch.Tensor` etc)
            >>> ...
            >>> return data_dict

        :param index: index of the sample to return.

        :return: DataDict containing the sample.

        """

        return DataDict({key: None for key in self.data_definitions.keys()})

    def add_ctrl(self, seq, ctrl, pos):
        """
        Adds control channels to a sequence.
        """
        return np.insert(seq, pos, ctrl, axis=-1)

    def augment(self, seq, markers, ctrl_start=None,
                add_marker_data=False, add_marker_dummy=True):
        """
        Creates augmented sequence as well as end marker and a dummy sequence.
        """
        ctrl_data, ctrl_dummy, pos = markers

        w = self.add_ctrl(seq, ctrl_data, pos)
        start = self.add_ctrl(
            np.zeros((seq.shape[0], 1, seq.shape[2])), ctrl_start, pos)
        if add_marker_data:
            w = np.concatenate((start, w), axis=1)

        start_dummy = self.add_ctrl(
            np.zeros((seq.shape[0], 1, seq.shape[2])), ctrl_dummy, pos)
        ctrl_data_select = np.ones(len(ctrl_data))
        dummy = self.add_ctrl(np.zeros_like(seq), ctrl_data_select, pos)

        if add_marker_dummy:
            dummy = np.concatenate((start_dummy, dummy), axis=1)

        return [w, dummy]

    def add_statistics(self, stat_col):
        """
        Add accuracy, seq_length and num_subsequences statistics to collector.

        :param stat_col: Statistics collector.

        """
        stat_col.add_statistic('acc', '{:12.10f}')
        stat_col.add_statistic('seq_length', '{:d}')
        #stat_col.add_statistic('num_subseq', '{:d}')
        stat_col.add_statistic('max_seq_length', '{:d}')

    def collect_statistics(self, stat_col, data_dict, logits):
        """
        Collects accuracy, seq_length and num_subsequences.

        :param stat_col: Statistics collector.
        :param data_dict: DataDict({'sequences', 'sequences_length', 'targets', 'mask', 'num_subsequences'}).
        :param logits: Logits being output of the model.

        """
        stat_col['acc'] = self.calculate_accuracy(data_dict, logits)
        stat_col['seq_length'] = data_dict['sequences_length']
        #stat_col['num_subseq'] = data_dict['num_subsequences']
        stat_col['max_seq_length'] = self.max_sequence_length

    def show_sample(self, data_dict, sample_number=0):
        """
        Shows the sample (both input and target sequences) using matplotlib.
        Elementary visualization.

        :param data_dict: DataDict({'sequences', 'sequences_length', 'targets', 'mask', 'num_subsequences'}).

        :param sample_number: Number of sample in a batch (DEFAULT: 0)

        """

        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        # Generate "canvas".
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False, gridspec_kw={
            'width_ratios': [data_dict['sequences'].shape[1]], 'height_ratios': [10, 10, 1]})
        # Set ticks.
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax3.yaxis.set_major_locator(ticker.NullLocator())

        # Set labels.
        ax1.set_title('Inputs')
        ax1.set_ylabel('Control/Data bits')
        ax2.set_title('Targets')
        ax2.set_ylabel('Data bits')
        ax3.set_title('Target mask')
        ax3.set_ylabel('Mask bit')
        ax3.set_xlabel('Item number', fontname='Times New Roman', fontsize=13)

        # print data
        print("\ninputs:", data_dict['sequences'][sample_number, :, :])
        print("\ntargets:", data_dict['targets'][sample_number, :, :])
        print("\nmask:", data_dict['mask'][sample_number:sample_number + 1, :])
        print("\nseq_length:", data_dict['sequences_length'])
        print("\nnum_subsequences:", data_dict['num_subsequences'])

        # show data.
        ax1.imshow(np.transpose(data_dict['sequences'][sample_number, :, :], [
                   1, 0]), interpolation='nearest', aspect='auto')
        ax2.imshow(np.transpose(data_dict['targets'][sample_number, :, :], [
                   1, 0]), interpolation='nearest', aspect='auto')
        ax3.imshow(data_dict['mask'][sample_number:sample_number + 1, :], interpolation='nearest', aspect='auto')
        # Plot!
        plt.tight_layout()
        plt.show()

    def curriculum_learning_update_params(self, episode):
        """
        Updates problem parameters according to curriculum learning. In the
        case of algorithmic sequential problems it updates the max sequence
        length, depending on configuration parameters.

        :param episode: Number of the current episode.
        :returns: Boolean informing whether curriculum learning is finished (or wasn't active at all).

        """
        # Curriculum learning stop condition.
        curric_done = True
        try:
            # Read curriculum learning parameters.
            max_max_length = self.params['max_sequence_length']
            interval = self.curriculum_params['interval']
            initial_max_sequence_length = self.curriculum_params['initial_max_sequence_length']

            if self.curriculum_params['interval'] > 0:
                # Curriculum learning goes from the initial max length to the
                # max length in steps of size 1
                max_length = initial_max_sequence_length + \
                    (episode // interval)
                if max_length >= max_max_length:
                    max_length = max_max_length
                else:
                    curric_done = False
                # Change max length.
                self.max_sequence_length = max_length
        except KeyError:
            pass
        # Return information whether we finished CL (i.e. reached max sequence
        # length).
        return curric_done


if __name__ == '__main__':

    from utils.param_interface import ParamInterface
    params = ParamInterface()
    params.add_custom_params({'control_bits': 2,
                              'data_bits': 8,
                              'min_sequence_length': 1,
                              'max_sequence_length': 10})

    sample = AlgorithmicSeqToSeqProblem(params)[0]
    # equivalent to ImageTextToClassProblem(params={}).__getitem__(index=0)

    print(repr(sample))
