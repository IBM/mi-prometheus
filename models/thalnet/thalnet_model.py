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

"""thalnet_model: Main class of the ThalNet paper: https://arxiv.org/abs/1706.05744. It calls the ThalNet cell on each word of the input"""

__author__ = "Younes Bouhadjar"

import torch
import logging
import numpy as np

from models.sequential_model import SequentialModel
from misc.app_state import AppState
from models.thalnet.thalnet_cell import ThalNetCell


class ThalNetModel(SequentialModel):
    """
    ThalNet model consists of recurrent neural modules that send features
    through a routing center, it was proposed in the following paper
    https://arxiv.org/pdf/1706.05744.pdf.
    """

    def __init__(self, params):
        """
        Constructor of the ThalNetModel.

        :param params: Parameters read from configuration file.

        """
        # Call base class initialization.
        super(ThalNetModel, self).__init__(params)

        self.context_input_size = params['context_input_size']
        self.input_size = params['input_size']
        self.output_size = params['output_size']
        self.center_size = params['num_modules'] * \
            params['center_size_per_module']
        self.center_size_per_module = params['center_size_per_module']
        self.num_modules = params['num_modules']
        self.output_center_size = self.output_size + self.center_size_per_module

        # This is for the time plot
        self.cell_state_history = None

        # Create the DWM components
        self.ThalnetCell = ThalNetCell(
            self.input_size,
            self.output_size,
            self.context_input_size,
            self.center_size_per_module,
            self.num_modules)

    def forward(self, data_tuple):  # x : batch_size, seq_len, input_size
        """
        Forward run of the ThalNetModel model.

        :param data_tuple: (inputs [batch_size, sequence_length, input_size], targets[batch_size, sequence_length, output_size])

        :returns: output: prediction [batch_size, sequence_length, output_size]

        """
        (inputs, _) = data_tuple

        if self.app_state.visualize:
            self.cell_state_history = []

        output = None
        batch_size = inputs.size(0)
        seq_length = inputs.size(-2)

        # init state
        cell_state = self.ThalnetCell.init_state(batch_size)
        for j in range(seq_length):
            output_cell, cell_state = self.ThalnetCell(
                inputs[..., j, :], cell_state)

            if output_cell is None:
                continue

            output_cell = output_cell[..., None, :]
            if output is None:
                output = output_cell

            # concatenate output
            else:
                output = torch.cat([output, output_cell], dim=-2)

            # This is for the time plot
            if self.app_state.visualize:
                self.cell_state_history.append(
                    [cell_state[i][0].detach().numpy()
                     for i in range(self.num_modules)] +
                    [cell_state[i][1].hidden_state.detach().numpy()
                     for i in range(self.num_modules)])

        return output

    def generate_figure_layout(self):
        from matplotlib.figure import Figure
        import matplotlib.ticker as ticker
        from matplotlib import rc
        import matplotlib.gridspec as gridspec

        # Change fonts globally - for all figures/subsplots at once.
        rc('font', **{'family': 'Times New Roman'})

        # Prepare "generic figure template".
        # Create figure object.
        fig = Figure()

        # Create a specific grid for NTM .
        gs = gridspec.GridSpec(4, 3)

        # modules & centers subplots
        ax_center = [fig.add_subplot(gs[i, 0])
                     for i in range(self.num_modules)]
        ax_module = [fig.add_subplot(gs[i, 1])
                     for i in range(self.num_modules)]  #

        # inputs & prediction subplot
        ax_inputs = fig.add_subplot(gs[0, 2])
        ax_pred = fig.add_subplot(gs[2, 2])

        # Set ticks - for bit axes only (for now).
        ax_inputs.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_inputs.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax_pred.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_pred.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Set labels.
        ax_inputs.set_title('Inputs')
        ax_inputs.set_ylabel('num_row')
        ax_inputs.set_xlabel('num_columns')

        ax_pred.set_title('Prediction')
        ax_pred.set_xlabel('num_classes')

        # centers
        ax_center[0].set_title('center states')
        ax_center[3].set_xlabel('iteration')
        ax_center[0].set_ylabel('center size')
        ax_center[1].set_ylabel('center size')
        ax_center[2].set_ylabel('center size')
        ax_center[3].set_ylabel('center size')

        # modules
        ax_module[0].set_title('module states')
        ax_module[3].set_xlabel('iteration')
        ax_module[0].set_ylabel('module state size')
        ax_module[1].set_ylabel('module state size')
        ax_module[2].set_ylabel('module state size')
        ax_module[3].set_ylabel('module state size')

        # Return figure.
        return fig

    def plot(self, data_tuple, logits, sample_number=0):
        # Check if we are supposed to visualize at all.
        if not self.app_state.visualize:
            return False

        # Initialize timePlot window - if required.
        if self.plotWindow is None:
            from misc.time_plot import TimePlot
            self.plotWindow = TimePlot()

        (inputs, _) = data_tuple
        inputs = inputs.cpu().detach().numpy()
        predictions_seq = logits.cpu().detach().numpy()

        input_seq = inputs[sample_number, 0] if len(
            inputs.shape) == 4 else inputs[sample_number]

        # Create figure template.
        fig = self.generate_figure_layout()
        # Get axes that artists will draw on.

        # Set intial values of displayed  inputs, targets and predictions -
        # simply zeros.
        inputs_displayed = np.zeros(input_seq.shape)

        # Define Modules
        module_state_displayed_1 = np.zeros(
            (self.cell_state_history[0][4].shape[-1], input_seq.shape[-2]))
        module_state_displayed_2 = np.zeros(
            (self.cell_state_history[0][4].shape[-1], input_seq.shape[-2]))
        module_state_displayed_3 = np.zeros(
            (self.cell_state_history[0][4].shape[-1], input_seq.shape[-2]))
        module_state_displayed_4 = np.zeros(
            (self.cell_state_history[0][-1].shape[-1], input_seq.shape[-2]))

        # Define centers
        center_state_displayed_1 = np.zeros(
            (self.cell_state_history[0][0].shape[-1], input_seq.shape[-2]))
        center_state_displayed_2 = np.zeros(
            (self.cell_state_history[0][0].shape[-1], input_seq.shape[-2]))
        center_state_displayed_3 = np.zeros(
            (self.cell_state_history[0][0].shape[-1], input_seq.shape[-2]))
        center_state_displayed_4 = np.zeros(
            (self.cell_state_history[0][0].shape[-1], input_seq.shape[-2]))

        #modules_plot = [module_state_displayed for _ in range(self.num_modules)]
        #center_plot = [center_state_displayed for _ in range(self.num_modules)]

        # Set initial values of memory and attentions.
        # Unpack initial state.

        # Log sequence length - so the user can understand what is going on.
        logger = logging.getLogger('ModelBase')
        logger.info(
            "Generating dynamic visualization of {} figures, please wait...".format(
                input_seq.shape[0]))

        # Create frames - a list of lists, where each row is a list of artists
        # used to draw a given frame.
        frames = []

        for i, (input_element, state_tuple) in enumerate(
                zip(input_seq, self.cell_state_history)):
            # Display information every 10% of figures.
            if (input_seq.shape[0] > 10) and (i %
                                              (input_seq.shape[0] // 10) == 0):
                logger.info(
                    "Generating figure {}/{}".format(i, input_seq.shape[0]))

            # Update displayed values on adequate positions.
            inputs_displayed[i, :] = input_element

            # Create "Artists" drawing data on "ImageAxes".
            artists = [None] * len(fig.axes)

            # centers state
            center_state_displayed_1[:, i] = state_tuple[0][sample_number, :]
            entity = fig.axes[0]
            artists[0] = entity.imshow(
                center_state_displayed_1,
                interpolation='nearest',
                aspect='auto')

            center_state_displayed_2[:, i] = state_tuple[1][sample_number, :]
            entity = fig.axes[1]
            artists[1] = entity.imshow(
                center_state_displayed_2,
                interpolation='nearest',
                aspect='auto')

            center_state_displayed_3[:, i] = state_tuple[2][sample_number, :]
            entity = fig.axes[2]
            artists[2] = entity.imshow(
                center_state_displayed_3,
                interpolation='nearest',
                aspect='auto')

            center_state_displayed_4[:, i] = state_tuple[3][sample_number, :]
            entity = fig.axes[3]
            artists[3] = entity.imshow(
                center_state_displayed_4,
                interpolation='nearest',
                aspect='auto')

            # module state
            module_state_displayed_1[:, i] = state_tuple[4][sample_number, :]
            entity = fig.axes[4]
            artists[4] = entity.imshow(
                module_state_displayed_1,
                interpolation='nearest',
                aspect='auto')

            module_state_displayed_2[:, i] = state_tuple[5][sample_number, :]
            entity = fig.axes[5]
            artists[5] = entity.imshow(
                module_state_displayed_2,
                interpolation='nearest',
                aspect='auto')

            module_state_displayed_3[:, i] = state_tuple[6][sample_number, :]
            entity = fig.axes[6]
            artists[6] = entity.imshow(
                module_state_displayed_3,
                interpolation='nearest',
                aspect='auto')

            module_state_displayed_4[:, i] = state_tuple[7][sample_number, :]
            entity = fig.axes[7]
            artists[7] = entity.imshow(
                module_state_displayed_4,
                interpolation='nearest',
                aspect='auto')

            # h = 0
            # for j, state in enumerate(state_tuple):
            #     # Get attention of head 0.
            #
            #     # "Show" data on "axes".
            #     entity = fig.axes[j]
            #     if self.num_modules <= h < 2 * self.num_modules :
            #         modules_plot[j - self.num_modules][:, i] = state[sample_number, :]
            #         artists[j] = entity.imshow(modules_plot[j - self.num_modules], interpolation='nearest', aspect='auto')
            #
            #     else:
            #         center_plot[j][:, i] = state[sample_number, :]
            #         artists[j] = entity.imshow(center_plot[j], interpolation='nearest', aspect='auto')
            #
            #     h += 1

            entity = fig.axes[2 * self.num_modules]
            artists[2 * self.num_modules] = entity.imshow(
                inputs_displayed, interpolation='nearest', aspect='auto')

            entity = fig.axes[2 * self.num_modules + 1]
            artists[2 *
                    self.num_modules +
                    1] = entity.imshow(predictions_seq[0, -
                                                       1, None], interpolation='nearest', aspect='auto')

            # Add "frame".
            frames.append(artists)

        # print("--- %s seconds ---" % (time.time() - start_time))
        # Update time plot fir generated list of figures.
        self.plotWindow.update(fig, frames)
        return self.plotWindow.is_closed


if __name__ == "__main__":
    input_size = 28
    params = {
        'context_input_size': 32,
        'input_size': input_size,
        'output_size': 10,
        'center_size': 1,
        'center_size_per_module': 32,
        'num_modules': 4}

    # Initialize the application state singleton.
    app_state = AppState()
    app_state.visualize = True

    model = ThalNetModel(params)

    seq_length = 10
    batch_size = 2

    # Check for different seq_lengts and batch_sizes.
    for i in range(1):
        # Create random Tensors to hold inputs and outputs
        x = torch.randn(batch_size, 1, input_size, input_size)
        logits = torch.randn(batch_size, 1, params['output_size'])
        y = x
        data_tuple = (x, y)

        # Test forward pass.
        y_pred = model(data_tuple)

        if model.plot(data_tuple, logits):
            break

        # Change batch size and seq_length.
        seq_length = seq_length + 1
        batch_size = batch_size + 1
