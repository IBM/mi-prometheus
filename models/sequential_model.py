#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sequential_model.py: contains base model for all sequential models"""
__author__ = "Tomasz Kornuta"

import numpy as np
import logging
import torch

from misc.app_state import AppState
from models.model import Model
from problems.problem import DataTuple


class SequentialModel(Model):
    """
    Class representing base class for all sequential models.

    Provides basic plotting functionality.

    """

    def __init__(self, params):
        """
        Initializes application state and sets plot if visualization flag is
        turned on.

        :param params: Parameters read from configuration file.

        """
        super(SequentialModel, self).__init__(params)

    def plot(self, data_tuple, predictions, sample_number=0):
        """
        Creates a default interactive visualization, with a slider enabling to
        move forth and back along the time axis (iteration in a given episode).
        The default visualizatoin contains input, output and target sequences.
        For more model/problem dependent visualization please overwrite this
        method in the derived model class.

        :param data_tuple: Data tuple containing
           - input [BATCH_SIZE x SEQUENCE_LENGTH x INPUT_DATA_SIZE] and
           - target sequences  [BATCH_SIZE x SEQUENCE_LENGTH x OUTPUT_DATA_SIZE]
        :param predictions: Prediction sequence [BATCH_SIZE x SEQUENCE_LENGTH x OUTPUT_DATA_SIZE]
        :param sample_number: Number of sample in batch (DEFAULT: 0)

        """
        # Check if we are supposed to visualize at all.
        if not self.app_state.visualize:
            return False

        # Initialize timePlot window - if required.
        if self.plotWindow is None:
            from misc.time_plot import TimePlot
            self.plotWindow = TimePlot()

        from matplotlib.figure import Figure
        import matplotlib.ticker as ticker

        # Change fonts globally - for all figures/subsplots at once.
        #from matplotlib import rc
        #rc('font', **{'family': 'Times New Roman'})
        import matplotlib.pylab as pylab
        params = {
            # 'legend.fontsize': '28',
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
        inputs_seq = data_tuple.inputs[sample_number].cpu().detach().numpy()
        targets_seq = data_tuple.targets[sample_number].cpu().detach().numpy()
        predictions_seq = predictions[sample_number].cpu().detach().numpy()

        # Create empty matrices.
        x = np.transpose(np.zeros(inputs_seq.shape))
        y = np.transpose(np.zeros(predictions_seq.shape))
        z = np.transpose(np.zeros(targets_seq.shape))

        # Log sequence length - so the user can understand what is going on.
        logger = logging.getLogger('ModelBase')
        logger.info(
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
                logger.info(
                    "Generating figure {}/{}".format(i, inputs_seq.shape[0]))

            # Add words to adequate positions.
            x[:, i] = input_word
            y[:, i] = target_word
            z[:, i] = prediction_word

            # Create "Artists" drawing data on "ImageAxes".
            artists = [None] * len(fig.axes)

            # Tell artists what to do;)
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

        # Return True if user closed the window.
        return self.plotWindow.is_closed


if __name__ == '__main__':
    # Set logging level.
    logging.basicConfig(level=logging.DEBUG)

    # Set visualization.
    AppState().visualize = True

    # Test sequential model.
    test = SequentialModel()

    while True:
        # Generate new sequence.
        x = np.random.binomial(1, 0.5, (1, 8, 15))
        y = np.random.binomial(1, 0.5, (1, 8, 15))
        z = np.random.binomial(1, 0.5, (1, 8, 15))

        # Transform to PyTorch.
        x = torch.from_numpy(x).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.FloatTensor)
        z = torch.from_numpy(z).type(torch.FloatTensor)
        dt = DataTuple(x, y)
        # Plot it and check whether window was closed or not.
        if test.plot(dt, z):
            break
