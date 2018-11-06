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

"""dwm_model.py: Main class of the Differentiable Working Memory. It calls the DWM cell on each word of the input"""
__author__ = "Younes Bouhadjar, T.S. Jayram, Tomasz Kornuta"

import torch
import logging
import numpy as np

from miprometheus.models.sequential_model import SequentialModel
from miprometheus.models.dwm.dwm_cell import DWMCell


class DWM(SequentialModel):
    """
    Differentiable Working Memory (DWM), is a memory augmented neural network
    which emulates the human working memory.

    The DWM shows the same functional characteristics of working memory
    and robustly learns psychology-inspired tasks and converges faster
    than comparable state-of-the-art models

    """

    def __init__(self, params, problem_default_values_={}):
        """
        Constructor. Initializes parameters on the basis of dictionary passed
        as argument.

        :param params: Local view to the Parameter Regsitry ''model'' section.

        :param problem_default_values_: Dictionary containing key-values received from problem.

        """
        # Call base constructor. Sets up default values etc.
        super(DWM, self).__init__(params, problem_default_values_)
        # Model name.
        self.name = "Differentiable Working Memory (DWM)"

        # Parse default values received from problem and add them to registry.
        self.params.add_default_params({
            'input_item_size': problem_default_values_['input_item_size'],
            'output_item_size': problem_default_values_['output_item_size']
            })


        self.in_dim = params["input_item_size"]
        self.output_units = params['output_item_size']

        self.state_units = params["hidden_state_size"]

        self.num_heads = params["num_heads"]
        self.is_cam = params["use_content_addressing"]
        self.num_shift = params["shift_size"]
        self.M = params["memory_content_size"]
        self.memory_addresses_size = params["memory_addresses_size"]

        # This is for the time plot
        self.cell_state_history = None

        # Create the DWM components
        self.DWMCell = DWMCell(
            self.in_dim,
            self.output_units,
            self.state_units,
            self.num_heads,
            self.is_cam,
            self.num_shift,
            self.M)

    def forward(self, data_dict):
        """
        Forward function requires that the data_dict will contain at least "sequences"

        :param data_dict: DataDict containing at least:
            - "sequences": a tensor of input data of size [BATCH_SIZE x LENGTH_SIZE x INPUT_SIZE]

        :returns: output: logits which represent the prediction of DWM [batch, sequence_length, output_size]

        Example:

        >>> dwm = DWM(params)
        >>> inputs = torch.randn(5, 3, 10)
        >>> targets = torch.randn(5, 3, 20)
        >>> data_tuple = (inputs, targets)
        >>> output = dwm(data_tuple)

        """
         # Get dtype.
        #dtype = self.app_state.dtype

        # Unpack dict.
        inputs = data_dict['sequences']
        
        # Get batch size and seq length.
        batch_size = inputs.size(0)
        seq_length = inputs.size(-2)

        if self.app_state.visualize:
            self.cell_state_history = []

        output = None
        # TODO
        if len(inputs.size()) == 4:
            inputs = inputs[:, 0, :, :]

        # The length of the memory is set to be equal to the input length in
        # case ```self.memory_addresses_size == -1```
        if self.memory_addresses_size == -1:
            if seq_length < self.num_shift:
                # memory size can't be smaller than num_shift (see
                # circular_convolution implementation)
                memory_addresses_size = self.num_shift
            else:
                memory_addresses_size = seq_length  # a hack for now
        else:
            memory_addresses_size = self.memory_addresses_size

        # Init state
        cell_state = self.DWMCell.init_state(memory_addresses_size, batch_size)

        # loop over the different sequences
        for j in range(seq_length):
            output_cell, cell_state = self.DWMCell(
                inputs[..., j, :], cell_state)

            if output_cell is None:
                continue

            output_cell = output_cell[..., None, :]
            if output is None:
                output = output_cell

            # Concatenate output
            else:
                output = torch.cat([output, output_cell], dim=-2)

            # This is for the time plot
            if self.app_state.visualize:
                self.cell_state_history.append(
                    (cell_state.memory_state.detach().numpy(),
                     cell_state.interface_state.head_weight.detach().numpy(),
                     cell_state.interface_state.snapshot_weight.detach().numpy()))

        return output

    # Method to change memory size
    def set_memory_size(self, mem_size):
        self.memory_addresses_size = mem_size

    @staticmethod
    def generate_figure_layout():
        """
        DOCUMENTATION!!

        """
        import matplotlib

        # Prepare "generic figure template".
        # Create figure object.
        fig = matplotlib.pyplot.figure(figsize=(16, 9))
        # fig.tight_layout()
        fig.subplots_adjust(left=0.07, right=0.96, top=0.88, bottom=0.15)
        
        gs0 = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[5.0, 3.0])

        # Create a specific grid for DWM .
        gs00 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[0],
                                                width_ratios=[3.0, 4.0, 4.0])

        ax_memory = fig.add_subplot(gs00[:, 0])  # all rows, col 0
        ax_attention = fig.add_subplot(gs00[:, 1])  # all rows, col 2-3
        ax_bookmark = fig.add_subplot(gs00[:, 2])  # all rows, col 4-5

        gs01 = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[1],
                                                hspace=0.5,
                                                height_ratios=[1.0, 0.8, 0.8])
        ax_inputs = fig.add_subplot(gs01[0, :])  # row 0, span 2 columns
        ax_targets = fig.add_subplot(gs01[1, :])  # row 0, span 2 columns
        ax_predictions = fig.add_subplot(gs01[2, :])  # row 0, span 2 columns

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
        ax_bookmark.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_bookmark.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_attention.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_attention.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

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
        ax_attention.set_title('Head Attention')
        ax_attention.set_xlabel('Iteration')
        ax_bookmark.set_title('Bookmark Attention')
        ax_bookmark.set_xlabel('Iteration')

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
        predictions_seq = predictions[0].cpu().detach()
        #predictions_seq = torch.sigmoid(predictions_seq).numpy()

        # temporary for data with additional channel
        if len(inputs_seq.shape) == 3:
            inputs_seq = inputs_seq[0, :, :]

        # Create figure template.
        fig = self.generate_figure_layout()
        # Get axes that artists will draw on.
        (ax_memory, ax_attention, ax_bookmark, ax_inputs,
         ax_targets, ax_predictions) = fig.axes

        # Set intial values of displayed  inputs, targets and predictions -
        # simply zeros.
        inputs_displayed = np.transpose(np.zeros(inputs_seq.shape))
        targets_displayed = np.transpose(np.zeros(targets_seq.shape))
        predictions_displayed = np.transpose(np.zeros(predictions_seq.shape))

        head_attention_displayed = np.zeros(
            (self.cell_state_history[0][1].shape[-1], targets_seq.shape[0]))
        bookmark_attention_displayed = np.zeros(
            (self.cell_state_history[0][2].shape[-1], targets_seq.shape[0]))

        # Log sequence length - so the user can understand what is going on.
        logger = logging.getLogger('ModelBase')
        logger.info(
            "Generating dynamic visualization of {} figures, please wait...".format(
                inputs_seq.shape[0]))

        # Create frames - a list of lists, where each row is a list of artists
        # used to draw a given frame.
        frames = []

        for i, (input_element, target_element, prediction_element, (memory, wt, wt_d)) in enumerate(
                zip(inputs_seq, targets_seq, predictions_seq, self.cell_state_history)):
            # Display information every 10% of figures.
            if (inputs_seq.shape[0] > 10) and (i %
                                               (inputs_seq.shape[0] // 10) == 0):
                logger.info(
                    "Generating figure {}/{}".format(i, inputs_seq.shape[0]))

            # Update displayed values on adequate positions.
            inputs_displayed[:, i] = input_element
            targets_displayed[:, i] = target_element
            predictions_displayed[:, i] = prediction_element

            memory_displayed = np.clip(memory[0], -3.0, 3.0)
            head_attention_displayed[:, i] = wt[0, 0, :]
            bookmark_attention_displayed[:, i] = wt_d[0, 0, :]

            # Create "Artists" drawing data on "ImageAxes".
            artists = [None] * len(fig.axes)
            params = {'edgecolor': 'black', 'cmap':'inferno', 'linewidths': 1.4e-3}

            # Tell artists what to do;)
            artists[0] = ax_memory.pcolormesh(np.transpose(memory_displayed),
                                              vmin=-3.0, vmax=3.0, **params)
            artists[1] = ax_attention.pcolormesh(np.copy(head_attention_displayed),
                                                 vmin=0.0, vmax=1.0, **params)
            artists[2] = ax_bookmark.pcolormesh(np.copy(bookmark_attention_displayed),
                                                vmin=0.0, vmax=1.0, **params)
            artists[3] = ax_inputs.pcolormesh(np.copy(inputs_displayed),
                                              vmin=0.0, vmax=1.0, **params)
            artists[4] = ax_targets.pcolormesh(np.copy(targets_displayed),
                                               vmin=0.0, vmax=1.0, **params)
            artists[5] = ax_predictions.pcolormesh(np.copy(predictions_displayed),
                                                   vmin=0.0, vmax=1.0, **params)

            # Add "frame".
            frames.append(artists)

        # Plot figure and list of frames.

        self.plotWindow.update(fig, frames)
        return self.plotWindow.is_closed
