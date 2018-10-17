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
algorithmic_seq_to_seq_problem.py: abstract base class for algorithmic sequential problems.

"""
__author__ = "Tomasz Kornuta, Younes Bouhadjar, Vincent Marois"

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.data_dict import DataDict
from problems.seq_to_seq.seq_to_seq_problem import SeqToSeqProblem
from utils.loss.masked_bce_with_logits_loss import MaskedBCEWithLogitsLoss


class AlgorithmicSeqToSeqProblem(SeqToSeqProblem):
    """
    Base class for algorithmic sequential problems.

    Provides some basic features useful in all problems of such nature.

    """

    def __init__(self, params):
        """
        Initializes problem object. Calls base ``SeqToSeqProblem`` constructor.

        Sets ``nn.BCEWithLogitsLoss()`` as the default loss function.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).

        """
        # call base constructor
        super(AlgorithmicSeqToSeqProblem, self).__init__(params)

        # "Default" problem name.
        self.name = 'AlgorithmicSeqToSeqProblem'

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
        params.add_default_params({'bias': 0.5})
        self.bias = params['bias']

        # set default data_definitions dict
        self.data_definitions = {'sequences': {'size': [-1, -1, -1], 'type': [torch.Tensor]},
                                 'sequences_length': {'size': [-1], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1, -1, -1], 'type': [torch.Tensor]},
                                 'mask': {'size': [-1, -1], 'type': [torch.Tensor]},
                                 'num_subsequences': {'size': [-1], 'type': [torch.Tensor]},
                                 }

        self.default_values = {'input_size': self.control_bits + self.data_bits,
                               'output_size': self.data_bits # Valid for most tasks, overwritten by equality/symmetry.
                               }

        # Set the default size of the dataset.
        # TODO: Should derive the actual theoretical limit instead of an arbitrary limit.
        params.add_default_params({'size': 1000})
        # Read value from registry - if it was set in config file, it will override the above default value.
        self.length = params['size']

    def set_max_length(self, max_length):
        """ Sets maximum sequence lenth (property).

        :param max_length: Length to be saved as max.
        """
        self.max_sequence_length = max_length

    def curriculum_learning_initialize(self, curriculum_params):
        """
        Initializes curriculum learning - simply saves the curriculum params.

        .. note::

            This method can be overwritten in the derived classes.


        :param curriculum_params: Interface to parameters accessing curriculum learning view of the registry tree.
        """
        # Save params.
        self.curriculum_params = curriculum_params
        # Inform the user.
        epoch_size = self.get_epoch_size(self.params["batch_size"])
        self.logger.info("Initializing curriculum learning! Will activate when all samples are exhausted \
            (every {} episodes when using batch of size {})".format(epoch_size, self.params["batch_size"]))

    def curriculum_learning_update_params(self, episode):
        """
        Updates problem parameters according to curriculum learning. In the \
        case of algorithmic sequential problems, it updates the max sequence \
        length, depending on configuration parameters.

        :param episode: Number of the current episode.
        :type episode: int

        :return: Boolean informing whether curriculum learning is finished (or wasn't active at all).

        """
        # Curriculum learning stop condition.
        curric_done = True
        try:
            # Read curriculum learning parameters.
            max_max_length = self.params['max_sequence_length']
            initial_max_sequence_length = self.curriculum_params['initial_max_sequence_length']
            epoch_size = self.get_epoch_size(self.params["batch_size"])

            # Curriculum learning goes from the initial max length to the
            # max length in steps of size 1
            max_length = initial_max_sequence_length + \
                ((episode+1) // epoch_size)
            if max_length > max_max_length:
                max_length = max_max_length
            else:
                curric_done = False
            # Change max length.
            self.max_sequence_length = max_length
        except KeyError:
            pass
        # Return information whether we finished CL (i.e. reached max sequence length).
        return curric_done

    def calculate_accuracy(self, data_dict, logits):
        """
        Calculate accuracy equal to mean difference between outputs and targets.

        .. warning::

            Applies mask to both logits and targets.


        :param data_dict: DataDict({'sequences', 'sequences_length', 'targets', 'mask', 'num_subsequences'}).

        :param logits: Predictions of the model.
        :type logits: tensor

        :return: Accuracy.

        """
        # Check if mask should be is used - if so, apply.
        if self.use_mask:
            return self.loss_function.masked_accuracy(
                logits, data_dict['targets'], data_dict['mask'])
        else:
            return (1 - torch.abs(torch.round(F.sigmoid(logits)) - data_dict['targets'])).mean()

    def add_ctrl(self, seq, ctrl, pos):
        """
        Adds control channels to a sequence.

        :param seq: Sequence to which controls channel are added.
        :type seq: array_like

        :param ctrl: Elements to add
        :type ctrl: array_like

        :param: pos: Object that defines the index or indices before which ctrl is inserted.
        :type pos: int, slice or sequence of ints

        :return: updated sequence.


        """
        return np.insert(seq, pos, ctrl, axis=-1)

    def augment(self, seq, markers, ctrl_start=None,
                add_marker_data=False, add_marker_dummy=True):
        """
        Creates augmented sequence as well as end marker and a dummy sequence.

        :param seq: Sequence
        :type seq: array_like

        :param markers: (ctrl_data, ctrl_dummy, pos)
        :type markers: tuple

        :param ctrl_start:
        :type ctrl_start:

        :param add_marker_data: Whether to add a marker before the data
        :type add_marker_data: bool

        :param add_marker_dummy: Whether to add a marker before the dummy
        :type add_marker_dummy: bool

        :return: [augmented_sequence, dummy]

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
        Add accuracy, seq_length and max_seq_length statistics to a ``StatisticsCollector``.

        :param stat_col: Statistics collector.
        :type stat_col: ``StatisticsCollector``

        """
        # Add basic statistics.
        super(AlgorithmicSeqToSeqProblem, self).add_statistics(stat_col)

        stat_col.add_statistic('acc', '{:12.10f}')
        stat_col.add_statistic('seq_length', '{:d}')
        #stat_col.add_statistic('num_subseq', '{:d}')
        stat_col.add_statistic('max_seq_length', '{:d}')
        stat_col.add_statistic('batch_size', '{:06d}')

    def collect_statistics(self, stat_col, data_dict, logits):
        """
        Collects accuracy, seq_length and max_seq_length.

        :param stat_col: Statistics collector.
        :type stat_col: ``StatisticsCollector``

        :param data_dict: DataDict({'sequences', 'sequences_length', 'targets', 'mask', 'num_subsequences'}).
        :type data_dict: DataDict

        :param logits: Predictions of the model.
        :type logits: tensor

        """
        # Collect basic statistics.
        super(AlgorithmicSeqToSeqProblem, self).collect_statistics(stat_col, data_dict, logits)

        stat_col['acc'] = self.calculate_accuracy(data_dict, logits)
        stat_col['seq_length'] = data_dict['sequences_length']
        #stat_col['num_subseq'] = data_dict['num_subsequences']
        stat_col['max_seq_length'] = self.max_sequence_length
        stat_col['batch_size'] = logits.shape[0] # Batch major.

    def add_aggregators(self, stat_agg):
        """
        Adds problem-dependent statistical aggregators to ``StatisticsAggregator``.

        :param stat_agg: ``StatisticsAggregator``.

        """
        # Add basic aggregators.
        super(AlgorithmicSeqToSeqProblem, self).add_aggregators(stat_agg)

        stat_agg.add_aggregator('acc', '{:12.10f}')  # represents the average accuracy
        stat_agg.add_aggregator('acc_min', '{:12.10f}')
        stat_agg.add_aggregator('acc_max', '{:12.10f}')
        stat_agg.add_aggregator('acc_std', '{:12.10f}')
        stat_agg.add_aggregator('samples_aggregated', '{:06d}')

    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates the statistics collected by ``StatisticsCollector`` and adds the results to ``StatisticsAggregator``.

        :param stat_col: ``StatisticsCollector``.

        :param stat_agg: ``StatisticsAggregator``.

        """
        # Aggregate base statistics.
        super(AlgorithmicSeqToSeqProblem, self).aggregate_statistics(stat_col, stat_agg)

        stat_agg['acc_min'] = min(stat_col['acc'])
        stat_agg['acc_max'] = max(stat_col['acc'])
        stat_agg['acc'] = torch.mean(torch.tensor(stat_col['acc']))
        stat_agg['acc_std'] = 0.0 if len(stat_col['acc']) <= 1 else torch.std(torch.tensor(stat_col['acc']))
        stat_agg['samples_aggregated'] = sum(stat_col['batch_size'])

    def show_sample(self, data_dict, sample=0):
        """
        Shows the sample (both input and target sequences) using ``matplotlib``.
        Elementary visualization.

        :param data_dict: DataDict({'sequences', 'sequences_length', 'targets', 'mask', 'num_subsequences'}).
        :type data_dict: DataDict

        :param sample: Number of sample in a batch (Default: 0)
        :type sample: int

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
        #print("\ninputs:", data_dict['sequences'][sample, :, :])
        #print("\ntargets:", data_dict['targets'][sample, :, :])
        #print("\nmask:", data_dict['mask'][sample:sample + 1, :])
        #print("\nseq_length:", data_dict['sequences_length'])
        #print("\nnum_subsequences:", data_dict['num_subsequences'])

        # show data.
        ax1.imshow(np.transpose(data_dict['sequences'][sample, :, :], [
                   1, 0]), interpolation='nearest', aspect='auto')
        ax2.imshow(np.transpose(data_dict['targets'][sample, :, :], [
                   1, 0]), interpolation='nearest', aspect='auto')
        ax3.imshow(data_dict['mask'][sample:sample + 1, :], interpolation='nearest', aspect='auto')
        # Plot!
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    from utils.param_interface import ParamInterface
    params = ParamInterface()
    params.add_config_params({'control_bits': 2,
                              'data_bits': 8,
                              'min_sequence_length': 1,
                              'max_sequence_length': 10})

    sample = AlgorithmicSeqToSeqProblem(params)[0]
    # equivalent to ImageTextToClassProblem(params={}).__getitem__(index=0)

    print(repr(sample))
