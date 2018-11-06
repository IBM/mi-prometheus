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

"""ntm_model.py: pytorch module implementing Neural Turing Machine"""
__author__ = "Tomasz Kornuta"

import torch
import logging
import numpy as np

from miprometheus.utils.data_dict import DataDict
from miprometheus.models.sequential_model import SequentialModel
from miprometheus.models.ntm.ntm_cell import NTMCell

class NTM(SequentialModel):
    """
    Class representing the Neural Turing Machine module.
    """

    def __init__(self, params, problem_default_values_={}):
        """
        Constructor. Initializes parameters on the basis of dictionary passed
        as argument.

        :param params: Local view to the Parameter Regsitry ''model'' section.

        :param problem_default_values_: Dictionary containing key-values received from problem.

        """
        # Call base constructor. Sets up default values etc.
        super(NTM, self).__init__(params, problem_default_values_)
        # Model name.
        self.name = 'NTM'

        # Parse default values received from problem and add them to registry.
        self.params.add_default_params({
            'input_item_size': problem_default_values_['input_item_size'],
            'output_item_size': problem_default_values_['output_item_size']
            })

        # Parse parameters.
        # It is stored here, but will we used ONLY ONCE - for initialization of
        # memory in the forward() function.
        self.num_memory_addresses = params['memory']['num_addresses']
        self.num_memory_content_bits = params['memory']['num_content_bits']

        # Initialize recurrent NTM cell.
        self.ntm_cell = NTMCell(params)

        # Set different visualizations depending on the flags.
        try:
            if params['visualization_mode'] == 1:
                self.plot = self.plot_memory_attention_sequence
            elif params['visualization_mode'] == 2:
                self.plot = self.plot_memory_all_model_params_sequence
            # else: default visualization.
        except KeyError:
             # If the 'visualization_mode' key is not present, catch the exception and do nothing
             # I.e. show default vizualization.
            pass

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
        inputs_BxSxI = data_dict['sequences']
        
        # Get batch size.
        batch_size = inputs_BxSxI.size(0)

        # "Data-driven memory size".
        # Save as TEMPORAL VARIABLE!
        # (do not overwrite self.num_memory_addresses, which will cause problem with next batch!)
        if self.num_memory_addresses == -1:
            # Set equal to input sequence length.
            num_memory_addresses = inputs_BxSxI.size(1)
        else:
            num_memory_addresses = self.num_memory_addresses

        # Initialize memory [BATCH_SIZE x MEMORY_ADDRESSES x CONTENT_BITS]
        init_memory_BxAxC = torch.zeros(
            batch_size,
            num_memory_addresses,
            self.num_memory_content_bits).type(dtype)

        # Initialize 'zero' state.
        cell_state = self.ntm_cell.init_state(init_memory_BxAxC)

        # List of output logits [BATCH_SIZE x OUTPUT_SIZE] of length SEQ_LENGTH
        output_logits_BxO_S = []

        # Check if we want to collect cell history for the visualization
        # purposes.
        if self.app_state.visualize:
            self.cell_state_history = []
            self.cell_state_initial = cell_state

        # Divide sequence into chunks of size [BATCH_SIZE x INPUT_SIZE] and
        # process them one by one.
        for input_t_Bx1xI in inputs_BxSxI.chunk(inputs_BxSxI.size(1), dim=1):
            # Process one chunk.
            output_BxO, cell_state = self.ntm_cell(
                input_t_Bx1xI.squeeze(1), cell_state)
            # Append to list of logits.
            output_logits_BxO_S += [output_BxO]

            # Collect cell history - for the visualization purposes.
            if self.app_state.visualize:
                self.cell_state_history.append(cell_state)

        # Stack logits along time axis (1).
        output_logits_BxSxO = torch.stack(output_logits_BxO_S, 1)

        return output_logits_BxSxO

    def generate_memory_attention_figure_layout(self):
        """
        Creates a figure template for showing basic NTM attributes (write &
        write attentions), memory and sequence (inputs, predictions and
        targets).

        :returns: Matplot figure object.

        """
        import matplotlib
        from matplotlib.figure import Figure

        # Change fonts globally - for all figures/subsplots at once.
        matplotlib.rc('font', **{'family': 'Times New Roman'})

        # Prepare "generic figure template".
        # Create figure object.
        fig = Figure()

        # Create a specific grid for NTM .
        gs = matplotlib.gridspec.GridSpec(3, 7)

        # Memory
        ax_memory = fig.add_subplot(gs[:, 0])  # all rows, col 0
        ax_write_attention = fig.add_subplot(gs[:, 1:3])  # all rows, col 2-3
        ax_read_attention = fig.add_subplot(gs[:, 3:5])  # all rows, col 4-5

        ax_inputs = fig.add_subplot(gs[0, 5:])  # row 0, span 2 columns
        ax_targets = fig.add_subplot(gs[1, 5:])  # row 0, span 2 columns
        ax_predictions = fig.add_subplot(gs[2, 5:])  # row 0, span 2 columns

        # Set ticks - currently for all axes.
        for ax in fig.axes:
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

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

        ax_write_attention.set_title('Write Attention')
        ax_write_attention.set_xlabel('Iteration')
        ax_read_attention.set_title('Read Attention')
        ax_read_attention.set_xlabel('Iteration')

        fig.set_tight_layout(True)
        return fig

    def plot_memory_attention_sequence(
            self, data_dict, predictions, sample_number=0):
        """
        Creates list of figures used in interactive visualization, with a
        slider enabling to move forth and back along the time axis (iteration
        in a given episode). The visualization presents input, output and
        target sequences passed as input parameters. Additionally, it utilizes
        state tuples collected during the experiment for displaying the memory
        state, read and write attentions.

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
        # Create figure template.
        fig = self.generate_memory_attention_figure_layout()
        # Get axes that artists will draw on.
        (ax_memory, ax_write_attention, ax_read_attention,
         ax_inputs, ax_targets, ax_predictions) = fig.axes

        # Unpack data tuple.
        inputs_seq = data_dict["sequences"][sample_number].cpu().detach().numpy()
        targets_seq = data_dict["targets"][sample_number].cpu().detach().numpy()
        predictions_seq = predictions[sample_number].cpu().detach().numpy()

        # Set intial values of displayed  inputs, targets and predictions -
        # simply zeros.
        inputs_displayed = np.transpose(np.zeros(inputs_seq.shape))
        targets_displayed = np.transpose(np.zeros(targets_seq.shape))
        predictions_displayed = np.transpose(np.zeros(predictions_seq.shape))

        # Set initial values of memory and attentions.
        # Unpack initial state.
        (ctrl_state, interface_state, memory_state,
         read_vectors) = self.cell_state_initial

       # Initialize "empty" matrices.
        memory_displayed = memory_state[0]
        read0_attention_displayed = np.zeros(
            (memory_state.shape[1], targets_seq.shape[0]))
        write_attention_displayed = np.zeros(
            (memory_state.shape[1], targets_seq.shape[0]))

        # Log sequence length - so the user can understand what is going on.
        logger = logging.getLogger('ModelBase')
        logger.info(
            "Generating dynamic visualization of {} figures, please wait...".format(
                inputs_seq.shape[0]))

        # Create frames - a list of lists, where each row is a list of artists
        # used to draw a given frame.
        frames = []

        for i, (input_element, target_element, prediction_element, cell_state) in enumerate(
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

            # Unpack cell state.
            (ctrl_state, interface_state, memory_state, read_vectors) = cell_state
            (read_state_tuples, write_state_tuple) = interface_state
            (write_attention, write_similarity,
             write_gate, write_shift) = write_state_tuple
            (read_attentions, read_similarities, read_gates,
             read_shifts) = zip(*read_state_tuples)

            # Set variables.
            memory_displayed = memory_state[0].detach().numpy()
            # Get attention of head 0.
            read0_attention_displayed[:,
                                      i] = read_attentions[0][0][:,
                                                                 0].detach().numpy()
            write_attention_displayed[:,
                                      i] = write_attention[0][:,
                                                              0].detach().numpy()

            # Create "Artists" drawing data on "ImageAxes".
            artists = [None] * len(fig.axes)

           # "Show" data on "axes".
            artists[0] = ax_memory.imshow(
                memory_displayed, interpolation='nearest', aspect='auto')
            artists[1] = ax_read_attention.imshow(
                read0_attention_displayed, interpolation='nearest', aspect='auto')
            artists[2] = ax_write_attention.imshow(
                write_attention_displayed, interpolation='nearest', aspect='auto')
            artists[3] = ax_inputs.imshow(
                inputs_displayed, interpolation='nearest', aspect='auto')
            artists[4] = ax_targets.imshow(
                targets_displayed, interpolation='nearest', aspect='auto')
            artists[5] = ax_predictions.imshow(
                predictions_displayed, interpolation='nearest', aspect='auto')

            # Add "frame".
            frames.append(artists)

        # print("--- %s seconds ---" % (time.time() - start_time))
        # Plot figure and list of frames.

        self.plotWindow.update(fig, frames)
        return self.plotWindow.is_closed

    def generate_memory_all_model_params_figure_layout(self):
        """
        Creates a figure template for showing all NTM attributes (write & write
        attentions, gates, convolution masks), along with memory and sequence
        (inputs, predictions and targets).

        :returns: Matplot figure object.

        """
        import matplotlib
        from matplotlib.figure import Figure

        # Change fonts globally - for all figures/subsplots at once.
        matplotlib.rc('font', **{'family': 'Times New Roman'})

        # Prepare "generic figure template".
        # Create figure object.
        fig = Figure()
        #axes = fig.subplots(3, 1, sharex=True, sharey=False, gridspec_kw={'width_ratios': [input_seq.shape[0]]})

        # Create a specific grid for NTM .
        gs = matplotlib.gridspec.GridSpec(4, 7)

        # Memory
        ax_memory = fig.add_subplot(gs[1:, 0])  # all rows, col 0
        ax_write_gate = fig.add_subplot(gs[0, 1:2])
        ax_write_shift = fig.add_subplot(gs[0, 2:3])
        ax_write_attention = fig.add_subplot(gs[1:, 1:2])  # all rows, col 2-3
        ax_write_similarity = fig.add_subplot(gs[1:, 2:3])
        ax_read_gate = fig.add_subplot(gs[0, 3:4])
        ax_read_shift = fig.add_subplot(gs[0, 4:5])
        ax_read_attention = fig.add_subplot(gs[1:, 3:4])  # all rows, col 4-5
        ax_read_similarity = fig.add_subplot(gs[1:, 4:5])

        ax_inputs = fig.add_subplot(gs[1, 5:])  # row 0, span 2 columns
        ax_targets = fig.add_subplot(gs[2, 5:])  # row 0, span 2 columns
        ax_predictions = fig.add_subplot(gs[3, 5:])  # row 0, span 2 columns

        # Set ticks - currently for all axes.
        for ax in fig.axes:
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        # ... except gates - single bit.
        ax_write_gate.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax_read_gate.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

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

        for ax in [
                ax_write_gate,
                ax_write_shift,
                ax_write_attention,
                ax_write_similarity,
                ax_read_gate,
                ax_read_shift,
                ax_read_attention,
                ax_read_similarity]:
            ax.set_xlabel('Iteration')

        # Write head.
        ax_write_gate.set_title('Write Gate')
        ax_write_shift.set_title('Write Shift')
        ax_write_attention.set_title('Write Attention')
        ax_write_similarity.set_title('Write Similarity')

        # Read head.
        ax_read_gate.set_title('Read Gate')
        ax_read_shift.set_title('Read Shift')
        ax_read_attention.set_title('Read Attention')
        ax_read_similarity.set_title('Read Similarity')

        fig.set_tight_layout(True)
        return fig

    def plot_memory_all_model_params_sequence(
            self, data_dict, predictions, sample_number=0):
        """
        Creates list of figures used in interactive visualization, with a
        slider enabling to move forth and back along the time axis (iteration
        in a given episode). The visualization presents input, output and
        target sequences passed as input parameters. Additionally, it utilizes
        state tuples collected during the experiment for displaying the memory
        state, read and write attentions; and gating params.

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
        # Create figure template.
        fig = self.generate_memory_all_model_params_figure_layout()
        # Get axes that artists will draw on.
        (ax_memory,
         ax_write_gate,
         ax_write_shift,
         ax_write_attention,
         ax_write_similarity,
         ax_read_gate,
         ax_read_shift,
         ax_read_attention,
         ax_read_similarity,
         ax_inputs,
         ax_targets,
         ax_predictions) = fig.axes

        # Unpack data tuple.
        inputs_seq = data_dict["sequences"][sample_number].cpu().detach().numpy()
        targets_seq = data_dict["targets"][sample_number].cpu().detach().numpy()

        predictions_seq = predictions[sample_number].cpu().detach().numpy()

        # Set intial values of displayed  inputs, targets and predictions -
        # simply zeros.
        inputs_displayed = np.transpose(np.zeros(inputs_seq.shape))
        targets_displayed = np.transpose(np.zeros(targets_seq.shape))
        predictions_displayed = np.transpose(np.zeros(predictions_seq.shape))

        # Set initial values of memory and attentions.
        # Unpack initial state.
        (ctrl_state, interface_state, memory_state,
         read_vectors) = self.cell_state_initial
        (read_state_tuples, write_state_tuple) = interface_state
        (write_attention, write_similarity,
         write_gate, write_shift) = write_state_tuple

       # Initialize "empty" matrices.
        memory_displayed = memory_state[0]
        read0_attention_displayed = np.zeros(
            (memory_state.shape[1], targets_seq.shape[0]))
        read0_similarity_displayed = np.zeros(
            (memory_state.shape[1], targets_seq.shape[0]))
        read0_gate_displayed = np.zeros(
            (write_gate.shape[1], targets_seq.shape[0]))
        read0_shift_displayed = np.zeros(
            (write_shift.shape[1], targets_seq.shape[0]))

        write_attention_displayed = np.zeros(
            (memory_state.shape[1], targets_seq.shape[0]))
        write_similarity_displayed = np.zeros(
            (memory_state.shape[1], targets_seq.shape[0]))
        # Generally we can use write shapes as are the same.
        write_gate_displayed = np.zeros(
            (write_gate.shape[1], targets_seq.shape[0]))
        write_shift_displayed = np.zeros(
            (write_shift.shape[1], targets_seq.shape[0]))

        # Log sequence length - so the user can understand what is going on.
        logger = logging.getLogger('ModelBase')
        logger.info(
            "Generating dynamic visualization of {} figures, please wait...".format(
                inputs_seq.shape[0]))

        # Create frames - a list of lists, where each row is a list of artists
        # used to draw a given frame.
        frames = []

        for i, (input_element, target_element, prediction_element, cell_state) in enumerate(
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

            # Unpack cell state.
            (ctrl_state, interface_state, memory_state, read_vectors) = cell_state
            (read_state_tuples, write_state_tuple) = interface_state
            (write_attention, write_similarity,
             write_gate, write_shift) = write_state_tuple
            (read_attentions, read_similarities, read_gates,
             read_shifts) = zip(*read_state_tuples)

            # Set variables.
            memory_displayed = memory_state[0].detach().numpy()
            # Get params of read head 0.
            read0_attention_displayed[:,
                                      i] = read_attentions[0][0][:,
                                                                 0].detach().numpy()
            read0_similarity_displayed[:,
                                       i] = read_similarities[0][0][:,
                                                                    0].detach().numpy()
            read0_gate_displayed[:,
                                 i] = read_gates[0][0][:, 0].detach().numpy()
            read0_shift_displayed[:,
                                  i] = read_shifts[0][0][:, 0].detach().numpy()

            # Get params of write head
            write_attention_displayed[:,
                                      i] = write_attention[0][:,
                                                              0].detach().numpy()
            write_similarity_displayed[:,
                                       i] = write_similarity[0][:,
                                                                0].detach().numpy()
            write_gate_displayed[:, i] = write_gate[0][:, 0].detach().numpy()
            write_shift_displayed[:, i] = write_shift[0][:, 0].detach().numpy()

            # Create "Artists" drawing data on "ImageAxes".
            artists = [None] * len(fig.axes)

            # "Show" data on "axes".
            artists[0] = ax_memory.imshow(
                memory_displayed, interpolation='nearest', aspect='auto')

            # Read head.
            artists[1] = ax_read_attention.imshow(
                read0_attention_displayed, interpolation='nearest', aspect='auto')
            artists[2] = ax_read_similarity.imshow(
                read0_similarity_displayed, interpolation='nearest', aspect='auto')
            artists[3] = ax_read_gate.imshow(
                read0_gate_displayed, interpolation='nearest', aspect='auto')
            artists[4] = ax_read_shift.imshow(
                read0_shift_displayed, interpolation='nearest', aspect='auto')

            # Write head.
            artists[5] = ax_write_attention.imshow(
                write_attention_displayed, interpolation='nearest', aspect='auto')
            artists[6] = ax_write_similarity.imshow(
                write_similarity_displayed, interpolation='nearest', aspect='auto')
            artists[7] = ax_write_gate.imshow(
                write_gate_displayed, interpolation='nearest', aspect='auto')
            artists[8] = ax_write_shift.imshow(
                write_shift_displayed, interpolation='nearest', aspect='auto')

            # "Default data".
            artists[9] = ax_inputs.imshow(
                inputs_displayed, interpolation='nearest', aspect='auto')
            artists[10] = ax_targets.imshow(
                targets_displayed, interpolation='nearest', aspect='auto')
            artists[11] = ax_predictions.imshow(
                predictions_displayed, interpolation='nearest', aspect='auto')

            # Add "frame".
            frames.append(artists)

        # print("--- %s seconds ---" % (time.time() - start_time))
        # Plot figure and list of frames.

        self.plotWindow.update(fig, frames)
        return self.plotWindow.is_closed


if __name__ == "__main__":
    # Set logging level.
    logger = logging.getLogger('NTM-Module')
    logging.basicConfig(level=logging.DEBUG)

    # Set visualization.
    from miprometheus.utils.app_state import AppState
    AppState().visualize = True

    # "Loaded parameters".
    from miprometheus.utils.param_interface import ParamInterface
    params = ParamInterface()
    params.add_default_params({
              # controller parameters
              'controller': {'name': 'GRUController', 'hidden_state_size': 5, 'num_layers': 1, 'non_linearity': 'none', 'output_size': 5},
              # interface parameters
              'interface': {'num_read_heads': 2, 'shift_size': 3},
              # memory parameters
              'memory': {'num_addresses': 4, 'num_content_bits': 7},
              'visualization_mode': 2
              })
    logger.debug("params: {}".format(params))

    num_control_bits= 3
    num_data_bits = 8
    seq_length = 1
    batch_size = 2

    # "Default values from problem".
    problem_default_values = {
        'input_item_size': num_control_bits + num_data_bits,
        'output_item_size': num_data_bits,
        'store_bit': 0,
        'recall_bit': 1
        }
    input_size = problem_default_values['input_item_size']
    output_size = problem_default_values['output_item_size']

    # Construct our model by instantiating the class defined above
    model = NTM(params, problem_default_values)

    # Check for different seq_lengts and batch_sizes.
    for i in range(2):
        # Create random Tensors to hold inputs and outputs
        x = torch.randn(batch_size, seq_length, input_size)
        y = torch.randn(batch_size, seq_length, output_size)

        dt = DataDict({'sequences': x, 'targets': y})

        # Test forward pass.
        logger.info("------- forward -------")
        y_pred = model(dt)

        logger.info("------- result -------")
        logger.info("input {}:\n {}".format(x.size(), x))
        logger.info("target.size():\n {}".format(y.size()))
        logger.info("prediction {}:\n {}".format(y_pred.size(), y_pred))

        # Plot it and check whether window was closed or not.
        if model.plot(dt, y_pred):
            break

        # Change batch size and seq_length.
        seq_length = seq_length + 1
        batch_size = batch_size + 1
