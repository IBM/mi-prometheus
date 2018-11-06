#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""dnc_model.py: Main class of the Differentiable Neural Computer. It calls the DNC cell on each word of the input"""
__author__ = "Ryan L. McAvoy, Tomasz Kornuta"

import numpy as np
import torch

import logging
from miprometheus.models.sequential_model import SequentialModel
from miprometheus.models.dnc.dnc_cell import DNCCell


class DNC(SequentialModel):
    """
        Implementation of Differentiable Neural Computer (DNC)

        Graves, Alex, et al. "Hybrid computing using a neural network with dynamic external memory."
        Nature 538.7626 (2016): 471. doi:10.1038/nature20101
    """

    def __init__(self, params, problem_default_values_={}):
        """
        Constructor. Initializes parameters on the basis of dictionary passed
        as argument.

        :param params: Local view to the Parameter Regsitry ''model'' section.

        :param problem_default_values_: Dictionary containing key-values received from problem.

        """
        # Call base constructor. Sets up default values etc.
        super(DNC, self).__init__(params, problem_default_values_)
        # Model name.
        self.name = 'DNC'

        # Parse default values received from problem and add them to registry.
        self.params.add_default_params({
            'input_item_size': problem_default_values_['input_item_size'],
            'output_item_size': problem_default_values_['output_item_size']
            })

        self.output_units = params['output_item_size']


        self.memory_addresses_size = params["memory_addresses_size"]
        self.label = params["name"]
        self.cell_state_history = None

        # Number of read and write heads
        self._num_reads = params["num_reads"]
        self._num_writes = params["num_writes"]

        # Create the DNC components
        self.DNCCell = DNCCell(self.output_units, params)

    def forward(self, data_dict):
        """
        Forward function requires that the data_dict will contain at least "sequences"

        :param data_dict: DataDict containing at least:
            - "sequences": a tensor of input data of size [BATCH_SIZE x LENGTH_SIZE x INPUT_SIZE]

        :returns: Predictions (logits) being a tensor of size  [BATCH_SIZE x LENGTH_SIZE x OUTPUT_SIZE].

        """
         # Get dtype.
        dtype = self.app_state.dtype

        # Unpack dict.
        inputs = data_dict['sequences']
        
        # Get batch size and seq length.
        batch_size = inputs.size(0)
        seq_length = inputs.size(1)

        output = None

        if self.app_state.visualize:
            self.cell_state_history = []


        memory_addresses_size = self.memory_addresses_size

        # if memory size is not fixed, set it to the total input plus output
        # size
        if memory_addresses_size == -1:
            memory_addresses_size = seq_length

        # init state
        cell_state = self.DNCCell.init_state(memory_addresses_size, batch_size)

        #cell_state = self.init_state(memory_addresses_size)
        for j in range(seq_length):
            output_cell, cell_state = self.DNCCell(
                inputs[..., j, :], cell_state)

            if output_cell is None:
                continue

            output_cell = output_cell[..., None, :]
            if output is None:
                output = output_cell
            else:
                output = torch.cat([output, output_cell], dim=-2)

            # This is for the time plot
            if self.app_state.visualize:
                self.cell_state_history.append(
                    (cell_state.memory_state.detach().cpu().numpy(),
                     cell_state.int_init_state.usage.detach().cpu().numpy(),
                     cell_state.int_init_state.links.precedence_weights.detach().cpu().numpy(),
                     cell_state.int_init_state.read_weights.detach().cpu().numpy(),
                     cell_state.int_init_state.write_weights.detach().cpu().numpy()))

            # if self.plot_active:
            #    self.plot_memory_attention(output, cell_state)

        return output

    def plot_memory_attention(self, data_dict, predictions, sample_number=0):
        """
        Plots memory and attention TODO: fix.

        :param data_dict: DataDict containing at least:
            - "sequences": a tensor of input data of size [BATCH_SIZE x LENGTH_SIZE x INPUT_SIZE]
            - "targets": a tensor of targets of size  [BATCH_SIZE x SEQUENCE_LENGTH x OUTPUT_DATA_SIZE]

        :param predictions: Prediction sequence [BATCH_SIZE x SEQUENCE_LENGTH x OUTPUT_DATA_SIZE]

        :param sample_number: Number of sample in batch (DEFAULT: 0)

        """
        # plot attention/memory

        self.logger.warning("DNC 'plot_memory_attention' method not implemented!")
        #plot_memory_attention(output, states[2], states[1][0], states[1][1], states[1][2], self.label)

    def generate_figure_layout(self):
        """
        DOCUMENTATION!
        :return:
        """
        import matplotlib
        from matplotlib.figure import Figure

        # Change fonts globally - for all figures/subsplots at once.
        # from matplotlib import rc
        # rc('font', **{'family': 'Times New Roman'})
        params = {
            # 'legend.fontsize': '28',
            'axes.titlesize': 'large',
            'axes.labelsize': 'large',
            'xtick.labelsize': 'medium',
            'ytick.labelsize': 'medium'}
        matplotlib.pylab.rcParams.update(params)

        # Prepare "generic figure template".
        # Create figure object.
        fig = Figure()

        # Create a specific grid for DWM .
        gs = matplotlib.gridspec.GridSpec(3, 9)

        # Memory
        ax_memory = fig.add_subplot(gs[:, 0])  # all rows, col 0
        ax_read = fig.add_subplot(gs[:, 1:3])  # all rows, col 2-3
        ax_write = fig.add_subplot(gs[:, 3:5])  # all rows, col 4-5
        ax_usage = fig.add_subplot(gs[:, 5:7])  # all rows, col 4-5

        ax_inputs = fig.add_subplot(gs[0, 7:])  # row 0, span 2 columns
        ax_targets = fig.add_subplot(gs[1, 7:])  # row 0, span 2 columns
        ax_predictions = fig.add_subplot(gs[2, 7:])  # row 0, span 2 columns

        # Set ticks - for bit axes only (for now).
        ax_inputs.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_inputs.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_targets.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_targets.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_predictions.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True))
        ax_predictions.yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True))
        ax_memory.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_memory.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_read.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_read.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_write.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_write.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_usage.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_usage.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        # Set labels.
        ax_inputs.set_title('Inputs')
        ax_inputs.set_ylabel('Control/Data bits')
        ax_targets.set_title('Targets')
        ax_targets.set_ylabel('Data bits')
        ax_predictions.set_title('Predictions')
        ax_predictions.set_ylabel('Data bits')
        ax_predictions.set_xlabel('Item number/Iteration')

        ax_memory.set_title('Memory')
        ax_memory.set_ylabel('Memory Addresses')
        ax_memory.set_xlabel('Content bits')
        ax_read.set_title('Read attention')
        ax_read.set_xlabel('Iteration')
        ax_write.set_title('Write Attention')
        ax_write.set_xlabel('Iteration')
        ax_usage.set_title('Usage')
        ax_usage.set_xlabel('Iteration')

        fig.set_tight_layout(True)
        # gs.tight_layout(fig)
        # plt.tight_layout()
        #fig.subplots_adjust(left = 0)
        return fig

    def plot(self, data_dict, predictions, sample_number=0):
        """
        Interactive visualization, with a slider enabling to move forth and
        back along the time axis (iteration in a given episode).

        :param data_dict: DataDict containing at least:
            - "sequences": a tensor of input data of size [BATCH_SIZE x LENGTH_SIZE x INPUT_SIZE]
            - "targets": a tensor of targets of size  [BATCH_SIZE x SEQUENCE_LENGTH x OUTPUT_DATA_SIZE]

        :param predictions: Prediction sequence [BATCH_SIZE x SEQUENCE_LENGTH x OUTPUT_DATA_SIZE]
        :param sample_number: Number of sample in batch (DEFAULT: 0)

        """
        # Check if we are supposed to visualize at all.
        if not self.app_state.visualize:
            return False

        # Initialize timePlot window - if required.
        if self.plotWindow is None:
            from miprometheus.utils.time_plot import TimePlot
            self.plotWindow = TimePlot()

        # import time
        # start_time = time.time()
        inputs_seq = data_dict["sequences"][sample_number].cpu().detach().numpy()
        targets_seq = data_dict["targets"][sample_number].cpu().detach().numpy()
        predictions_seq = predictions[0].cpu().detach().numpy()

        # temporary for data with additional channel
        if len(inputs_seq.shape) == 3:
            inputs_seq = inputs_seq[0, :, :]

        # Create figure template.
        fig = self.generate_figure_layout()
        # Get axes that artists will draw on.
        (ax_memory, ax_read, ax_write, ax_usage,
         ax_inputs, ax_targets, ax_predictions) = fig.axes

        # Set intial values of displayed  inputs, targets and predictions -
        # simply zeros.
        inputs_displayed = np.transpose(np.zeros(inputs_seq.shape))
        targets_displayed = np.transpose(np.zeros(targets_seq.shape))
        predictions_displayed = np.transpose(np.zeros(predictions_seq.shape))

        head_attention_read = np.zeros(
            (self.cell_state_history[0][3].shape[-1], targets_seq.shape[0]))
        head_attention_write = np.zeros(
            (self.cell_state_history[0][4].shape[-1], targets_seq.shape[0]))
        usage_displayed = np.zeros(
            (self.cell_state_history[0][1].shape[-1], targets_seq.shape[0]))

        # Log sequence length - so the user can understand what is going on.
        logger = logging.getLogger('ModelBase')
        logger.info(
            "Generating dynamic visualization of {} figures, please wait...".format(
                inputs_seq.shape[0]))

        # Create frames - a list of lists, where each row is a list of artists
        # used to draw a given frame.
        frames = []

        for i, (input_element, target_element, prediction_element, (memory, usage, links, wt_r, wt_w)
                ) in enumerate(zip(inputs_seq, targets_seq, predictions_seq, self.cell_state_history)):
            # Display information every 10% of figures.
            if (inputs_seq.shape[0] > 10) and (i %
                                               (inputs_seq.shape[0] // 10) == 0):
                logger.info(
                    "Generating figure {}/{}".format(i, inputs_seq.shape[0]))

            # Update displayed values on adequate positions.
            inputs_displayed[:, i] = input_element
            targets_displayed[:, i] = target_element
            predictions_displayed[:, i] = prediction_element

            memory_displayed = memory[0]
            # Get attention of head 0.
            head_attention_read[:, i] = wt_r[0, 0, :]
            head_attention_write[:, i] = wt_w[0, 0, :]
            usage_displayed[:, i] = usage[0, :]

            # Create "Artists" drawing data on "ImageAxes".
            artists = [None] * len(fig.axes)

            # Tell artists what to do;)
            artists[0] = ax_memory.imshow(np.transpose(
                memory_displayed), interpolation='nearest', aspect='auto')
            artists[1] = ax_read.imshow(
                head_attention_read, interpolation='nearest', aspect='auto')
            artists[2] = ax_write.imshow(
                head_attention_write, interpolation='nearest', aspect='auto')
            artists[3] = ax_usage.imshow(
                usage_displayed, interpolation='nearest', aspect='auto')
            artists[4] = ax_inputs.imshow(
                inputs_displayed, interpolation='nearest', aspect='auto')
            artists[5] = ax_targets.imshow(
                targets_displayed, interpolation='nearest', aspect='auto')
            artists[6] = ax_predictions.imshow(
                predictions_displayed, interpolation='nearest', aspect='auto')

            # Add "frame".
            frames.append(artists)

        # print("--- %s seconds ---" % (time.time() - start_time))
        # Plot figure and list of frames.

        self.plotWindow.update(fig, frames)
        return self.plotWindow.is_closed
