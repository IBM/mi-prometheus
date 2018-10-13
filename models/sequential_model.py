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

"""sequential_model.py: contains base model for all sequential models"""
__author__ = "Tomasz Kornuta, Vincent Marois"

import torch
import numpy as np

from models.model import Model
from problems.problem import DataDict


class SequentialModel(Model):
    """
    Class representing base class for all Sequential Models.

    Inherits from models.model.Model as most features are the same.

    Should be derived by all sequential models.

    """

    def __init__(self, params, problem_default_values_={}):
        """
        Mostly calls the base ``models.model.Model`` constructor.

        Specifies a better structure for ``self.data_definitions``.

        :param params: Parameters read from configuration ``.yaml`` file.

        :param problem_default_values_: dict of parameters values coming from the problem class. One example of such \
        parameter value is the size of the vocabulary set in a translation problem.
        :type problem_default_values_: dict

        """
        super(SequentialModel, self).__init__(params, problem_default_values_=problem_default_values_)

        # "Default" model name.
        self.name = 'SequentialModel'

        # We can then define a dict that contains a description of the expected (and mandatory) inputs for this model.
        # This dict should be defined using self.params.
        self.data_definitions = {'sequences': {'size': [-1, -1, -1], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1, -1, -1], 'type': [torch.Tensor]}
                                 }

    def plot(self, data_dict, predictions, sample=0):
        """
        Creates a default interactive visualization, with a slider enabling to
        move forth and back along the time axis (iteration over the sequence elements in a given episode).
        The default visualization contains the input, output and target sequences.

        For a more model/problem - dependent visualization, please overwrite this
        method in the derived model class.

        :param data_dict: DataDict containing

           - input sequences: [BATCH_SIZE x SEQUENCE_LENGTH x INPUT_SIZE],
           - target sequences:  [BATCH_SIZE x SEQUENCE_LENGTH x OUTPUT_SIZE]


        :param predictions: Predicted sequences [BATCH_SIZE x SEQUENCE_LENGTH x OUTPUT_SIZE]
        :type predictions: torch.tensor

        :param sample: Number of sample in batch (default: 0)
        :type sample: int

        """
        # Check if we are supposed to visualize at all.
        if not self.app_state.visualize:
            return

        # Initialize timePlot window - if required.
        if self.plotWindow is None:
            from utils.time_plot import TimePlot
            self.plotWindow = TimePlot()

        from matplotlib.figure import Figure
        import matplotlib.ticker as ticker

        # Change fonts globally - for all figures/subsplots at once.
        # from matplotlib import rc
        # rc('font', **{'family': 'Times New Roman'})
        import matplotlib.pylab as pylab
        params = {# 'legend.fontsize': '28',
                  'axes.titlesize': 'large',
                  'axes.labelsize': 'large',
                  'xtick.labelsize': 'medium',
                  'ytick.labelsize': 'medium'}
        pylab.rcParams.update(params)

        # Create a single "figure layout" for all displayed frames.
        fig = Figure()
        axes = fig.subplots(3, 1, sharex=True, sharey=False, gridspec_kw={
                            'width_ratios': [predictions.shape[0]]})

        # Set ticks.
        axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axes[0].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axes[1].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axes[2].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Set labels.
        axes[0].set_title('Inputs')
        axes[0].set_ylabel('Control/Data bits')
        axes[1].set_title('Targets')
        axes[1].set_ylabel('Data bits')
        axes[2].set_title('Predictions')
        axes[2].set_ylabel('Data bits')
        axes[2].set_xlabel('Item number')
        fig.set_tight_layout(True)

        # Detach a sample from batch and copy it to CPU.
        inputs_seq = data_dict['sequences'][sample].cpu().detach().numpy()
        targets_seq = data_dict['targets'][sample].cpu().detach().numpy()
        predictions_seq = predictions[sample].cpu().detach().numpy()

        # Create empty matrices.
        x = np.transpose(np.zeros(inputs_seq.shape))
        y = np.transpose(np.zeros(predictions_seq.shape))
        z = np.transpose(np.zeros(targets_seq.shape))

        # Log sequence length - so the user can understand what is going on.
        self.logger.info(
            "Generating dynamic visualization of {} figures, please wait...".format(
                inputs_seq.shape[0]))

        # Create frames - a list of lists, where each row is a list of artists
        # used to draw a given frame.
        frames = []

        for i, (input_word, prediction_word, target_word) in enumerate(
                zip(inputs_seq, predictions_seq, targets_seq)):
            # Display information every 10% of figures.
            if (inputs_seq.shape[0] > 10) and (i %
                                               (inputs_seq.shape[0] // 10) == 0):
                self.logger.info(
                    "Generating figure {}/{}".format(i, inputs_seq.shape[0]))

            # Add words to adequate positions.
            x[:, i] = input_word
            y[:, i] = target_word
            z[:, i] = prediction_word

            # Create "Artists" drawing data on "ImageAxes".
            artists = [None] * len(fig.axes)

            # Tell artists what to do
            artists[0] = axes[0].imshow(
                x, interpolation='nearest', aspect='auto')
            artists[1] = axes[1].imshow(
                y, interpolation='nearest', aspect='auto')
            artists[2] = axes[2].imshow(
                z, interpolation='nearest', aspect='auto')

            # Add "frame".
            frames.append(artists)

        # Plot figure and list of frames.
        self.plotWindow.update(fig, frames)


if __name__ == '__main__':
    """Unit test of the SequentialModel"""

    from utils.param_interface import ParamInterface
    from utils.app_state import AppState

    # Set visualization.
    AppState().visualize = True

    # Test sequential model.
    sequential_model = SequentialModel(ParamInterface())

    # Set logging level.
    import logging
    logging.basicConfig(level=logging.DEBUG)

    while True:
        # Generate new sequence.
        x = np.random.binomial(1, 0.5, (1, 8, 15))
        y = np.random.binomial(1, 0.5, (1, 8, 15))
        z = np.random.binomial(1, 0.5, (1, 8, 15))

        # Transform to PyTorch.
        x = torch.from_numpy(x).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.FloatTensor)
        z = torch.from_numpy(z).type(torch.FloatTensor)
        dt = DataDict({'sequences': x, 'targets': y})

        # Plot it and check whether window was closed or not.
        if sequential_model.plot(dt, z):
            break
