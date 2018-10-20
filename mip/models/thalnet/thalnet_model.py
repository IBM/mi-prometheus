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
thalnet_model.py: Contains the Main class of the ThalNet model.


See the reference paper here: https://arxiv.org/pdf/1706.05744.pdf.
"""

__author__ = "Younes Bouhadjar & Vincent Marois"

import torch
import numpy as np

from mip.models.sequential_model import SequentialModel
from mip.models.thalnet.thalnet_cell import ThalNetCell


class ThalNetModel(SequentialModel):
    """
    ``ThalNet`` is a deep learning model inspired by neocortical communication \
    via the thalamus. This model consists of recurrent neural modules that send features \
    through a routing center, endowing the modules with the flexibility to share features \
    over multiple time steps.

    See the reference paper here: https://arxiv.org/pdf/1706.05744.pdf.

    .. warning:

        The reference paper indicates that the ``Thalnet`` model works on the Sequential MNIST problem. \
        This implementation does not for the moment, and has only been tested on the SerialRecall task so far.

        This should be adressed in a future release.

    """

    def __init__(self, params, problem_default_values_={}):
        """
        Constructor of the ``ThalNetModel``. Instantiates the ``ThalNetCell``.

        :param params: dictionary of parameters (read from the ``.yaml`` configuration file.)

        :param problem_default_values_: default values coming from the ``Problem`` class.
        :type problem_default_values_: dict

        """
        # Call base class initialization.
        super(ThalNetModel, self).__init__(params, problem_default_values_)

        # get the parameters values
        self.context_input_size = params['context_input_size']
        self.input_size = params['input_size']
        self.output_size = params['output_size']
        self.center_size = params['num_modules'] * params['center_size_per_module']
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

        # model name
        self.name = 'ThalNetModel'

        # Expected content of the inputs
        self.data_definitions = {'sequences': {'size': [-1, -1, -1], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1, -1, -1], 'type': [torch.Tensor]}
                                 }

    def forward(self, data_dict):  # x : batch_size, seq_len, input_size
        """
        Forward run of the ThalNetModel model.

        :param data_dict: DataDict({'sequences', **}) where 'sequences' is of shape \
         [batch_size, sequence_length, input_size]
        :type data_dict: utils.DataDict

        :returns: Predictions [batch_size, sequence_length, output_size]

        """
        inputs = data_dict['sequences']

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
        """
        Generate a figure layout which will be used in ``self.plot()``.

        :return: figure layout.

        """
        from matplotlib.figure import Figure
        import matplotlib.ticker as ticker
        from matplotlib import rc
        import matplotlib.gridspec as gridspec

        # Change fonts globally - for all figures/subsplots at once.
        rc('font', **{'family': 'Times New Roman'})

        # Prepare "generic figure template".
        # Create figure object.
        fig = Figure()

        # Create a specific grid
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

    def plot(self, data_dict, logits, sample=0):
        """
        Plots specific information on the model's behavior.

        :param data_dict: DataDict({'sequences', **})
        :type data_dict: utils.DataDict

        :param logits: Predictions of the model
        :type logits: torch.tensor

        :param sample: Index of the sample to visualize. Default to 0.
        :type sample: int

        :return: ``True`` if the user pressed stop, else ``False``.

        """
        # Check if we are supposed to visualize at all.
        if not self.app_state.visualize:
            return False

        # Initialize timePlot window - if required.
        if self.plotWindow is None:
            from mip.utils.time_plot import TimePlot
            self.plotWindow = TimePlot()

        inputs = data_dict['sequences']
        inputs = inputs.cpu().detach().numpy()
        predictions_seq = logits.cpu().detach().numpy()

        input_seq = inputs[sample, 0] if len(
            inputs.shape) == 4 else inputs[sample]

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

        # Set initial values of memory and attentions.
        # Unpack initial state.

        # Log sequence length - so the user can understand what is going on.

        self.logger.info(
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
                self.logger.info(
                    "Generating figure {}/{}".format(i, input_seq.shape[0]))

            # Update displayed values on adequate positions.
            inputs_displayed[i, :] = input_element

            # Create "Artists" drawing data on "ImageAxes".
            artists = [None] * len(fig.axes)

            # centers state
            center_state_displayed_1[:, i] = state_tuple[0][sample, :]
            entity = fig.axes[0]
            artists[0] = entity.imshow(
                center_state_displayed_1,
                interpolation='nearest',
                aspect='auto')

            center_state_displayed_2[:, i] = state_tuple[1][sample, :]
            entity = fig.axes[1]
            artists[1] = entity.imshow(
                center_state_displayed_2,
                interpolation='nearest',
                aspect='auto')

            center_state_displayed_3[:, i] = state_tuple[2][sample, :]
            entity = fig.axes[2]
            artists[2] = entity.imshow(
                center_state_displayed_3,
                interpolation='nearest',
                aspect='auto')

            center_state_displayed_4[:, i] = state_tuple[3][sample, :]
            entity = fig.axes[3]
            artists[3] = entity.imshow(
                center_state_displayed_4,
                interpolation='nearest',
                aspect='auto')

            # module state
            module_state_displayed_1[:, i] = state_tuple[4][sample, :]
            entity = fig.axes[4]
            artists[4] = entity.imshow(
                module_state_displayed_1,
                interpolation='nearest',
                aspect='auto')

            module_state_displayed_2[:, i] = state_tuple[5][sample, :]
            entity = fig.axes[5]
            artists[5] = entity.imshow(
                module_state_displayed_2,
                interpolation='nearest',
                aspect='auto')

            module_state_displayed_3[:, i] = state_tuple[6][sample, :]
            entity = fig.axes[6]
            artists[6] = entity.imshow(
                module_state_displayed_3,
                interpolation='nearest',
                aspect='auto')

            module_state_displayed_4[:, i] = state_tuple[7][sample, :]
            entity = fig.axes[7]
            artists[7] = entity.imshow(
                module_state_displayed_4,
                interpolation='nearest',
                aspect='auto')

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

        # Update time plot fir generated list of figures.
        self.plotWindow.update(fig, frames)

        return self.plotWindow.is_closed


if __name__ == "__main__":
    """ Unit test of the Thalnet Model.
    """
    input_size = 28
    params_dict = {
        'context_input_size': 32,
        'input_size': input_size,
        'output_size': 10,
        'center_size': 1,
        'center_size_per_module': 32,
        'num_modules': 4}

    # Initialize the application state singleton.
    from mip.utils.app_state import AppState
    from mip.utils.data_dict import DataDict
    app_state = AppState()
    app_state.visualize = True

    from mip.utils.param_interface import ParamInterface
    params = ParamInterface()
    params.add_config_params(params_dict)
    model = ThalNetModel(params)

    seq_length = 10
    batch_size = 2

    # Check for different seq_lengts and batch_sizes.
    for i in range(62):
        # Create random Tensors to hold inputs and outputs
        x = torch.randn(batch_size, 1, input_size, input_size)
        logits = torch.randn(batch_size, 1, params_dict['output_size'])
        y = x
        data_dict = DataDict({'sequences': x, 'targets': y})

        # Test forward pass.
        y_pred = model(data_dict)

        if model.plot(data_dict, logits):
            break

        # Change batch size and seq_length.
        seq_length = seq_length + 1
        batch_size = batch_size + 1
