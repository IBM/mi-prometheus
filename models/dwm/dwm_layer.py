import torch
import logging
import numpy as np
import io
import pickle

from torch import nn
from models.dwm.dwm_cell import DWMCell
from misc.app_state import AppState
from models.model_base import ModelBase

CUDA = False
dtype = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


class DWM(ModelBase, nn.Module):

    def __init__(self, params):
        """Initialize an DWM Layer.

        :param in_dim: input size.
        :param output_units: output size.
        :param state_units: state size.
        :param num_heads: number of heads.
        :param is_cam: is it content_addressable.
        :param num_shift: number of shifts of heads.
        :param M: Number of slots per address in the memory bank.
        """
        self.in_dim = params["control_bits"] + params["data_bits"]
        self.output_units = params["data_bits"]
        self.state_units =params["hidden_state_dim"]
        self.num_heads = params["num_heads"]
        self.is_cam = params["use_content_addressing"]
        self.num_shift = params["shift_size"]
        self.M = params["memory_content_size"]
        self.memory_addresses_size = params["memory_addresses_size"]
        self.label = params["name"]
        self.app_state = AppState()

        # This is for the time plot
        self.cell_state_history = None

        super(DWM, self).__init__()

        # Create the DWM components
        self.DWMCell = DWMCell(self.in_dim, self.output_units, self.state_units,
                               self.num_heads, self.is_cam, self.num_shift, self.M)

    def forward(self, data_tuple):       # x : batch_size, seq_len, input_size
        """
        Runs the DWM cell and plots if necessary
        
        :param x: input sequence  [BATCH_SIZE x seq_len x input_size ]
        :param state: Input hidden state  [BATCH_SIZE x state_size]
        :return: Tuple [output, hidden_state]
        """
        (inputs, targets) = data_tuple

        if self.app_state.visualize:
            self.cell_state_history = []

        output = None
        batch_size = inputs.size(0)
        seq_length = inputs.size(1)

        if self.memory_addresses_size == -1:
            if seq_length < self.num_shift:
                memory_addresses_size = self.num_shift  # memory size can't be smaller than num_shift (see circular_convolution implementation)
            else:
                memory_addresses_size = seq_length  # a hack for now
        else:
            memory_addresses_size = self.memory_addresses_size

        # init state
        cell_state = self.init_state(memory_addresses_size, batch_size)
        for j in range(seq_length):
            output_cell, cell_state = self.DWMCell(inputs[..., j, :], cell_state)

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
                self.cell_state_history.append((cell_state[3].detach().numpy(),
                                                cell_state[1].detach().numpy(),
                                                cell_state[2].detach().numpy()))

        return output

    def init_state(self, memory_addresses_size, batch_size):

        state = torch.ones((batch_size, self.state_units)).type(dtype)

        # initial attention  vector
        wt = torch.zeros((batch_size, self.num_heads, memory_addresses_size)).type(dtype)
        wt[:, 0:self.num_heads, 0] = 1.0

        # bookmark
        wt_dynamic = wt

        mem_t = (torch.ones((batch_size, self.M, memory_addresses_size)) * 0.01).type(dtype)

        states = [state, wt, wt_dynamic, mem_t]
        return states

    def set_memory_size(self, mem_size):
        self.memory_addresses_size = mem_size

    def pickle_figure_template(self):
        from matplotlib.figure import Figure
        import matplotlib.ticker as ticker
        from matplotlib import rc
        import matplotlib.gridspec as gridspec

        # Change fonts globally - for all figures/subsplots at once.
        rc('font', **{'family': 'Times New Roman'})

        # Prepare "generic figure template".
        # Create figure object.
        fig = Figure()
        # axes = fig.subplots(3, 1, sharex=True, sharey=False, gridspec_kw={'width_ratios': [input_seq.shape[0]]})

        # Create a specific grid for NTM .
        gs = gridspec.GridSpec(3, 7)

        # Memory
        ax_memory = fig.add_subplot(gs[:, 0])  # all rows, col 0
        ax_attention = fig.add_subplot(gs[:, 1:3])  # all rows, col 2-3
        ax_snapshot = fig.add_subplot(gs[:, 3:5])  # all rows, col 4-5

        ax_inputs = fig.add_subplot(gs[0, 5:])  # row 0, span 2 columns
        ax_targets = fig.add_subplot(gs[1, 5:])  # row 0, span 2 columns
        ax_predictions = fig.add_subplot(gs[2, 5:])  # row 0, span 2 columns

        # Set ticks - for bit axes only (for now).
        ax_inputs.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_inputs.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_targets.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_targets.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_predictions.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_predictions.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_memory.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_memory.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_snapshot.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_snapshot.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_attention.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_attention.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

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
        ax_attention.set_title('Head attention')
        ax_attention.set_xlabel('Iteration')
        ax_snapshot.set_title('Snapshot/Bookmark Attention')
        ax_snapshot.set_xlabel('Iteration')

        # Create buffer and pickle the figure.
        buf = io.BytesIO()
        pickle.dump(fig, buf)

        return buf

    def plot_sequence(self, output_seq, data_tuple):
        """ Creates a default interactive visualization, with a slider enabling to move forth and back along the time axis (iteration in a given episode).
        The default visualizatoin contains input, output and target sequences.
        For more model/problem dependent visualization please overwrite this method in the derived model class.
        """
        # import time
        # start_time = time.time()
        input_seq = data_tuple.inputs[0].cpu().detach().numpy()
        target_seq = data_tuple.targets[0].cpu().detach().numpy()
        output_seq = output_seq[0].cpu().detach().numpy()

        # Create figure template.
        buf = self.pickle_figure_template()

        # Set intial values of displayed  inputs, targets and predictions - simply zeros.
        inputs_displayed = np.transpose(np.zeros(input_seq.shape))
        targets_displayed = np.transpose(np.zeros(target_seq.shape))
        predictions_displayed = np.transpose(np.zeros(output_seq.shape))
        head_attention_displayed = np.zeros((self.cell_state_history[0][1].shape[-1], target_seq.shape[0]))
        snapshot_attention_displayed = np.zeros((self.cell_state_history[0][2].shape[-1], target_seq.shape[0]))

        # Set initial values of memory and attentions.
        # Unpack initial state.

        # Log sequence length - so the user can understand what is going on.
        logger = logging.getLogger('ModelBase')
        logger.info("Generating dynamic visualization of {} figures, please wait...".format(input_seq.shape[0]))
        # List of figures.
        figs = []
        for i, (input_element, output_elementd, target_element, (memory, wt, wt_d)) in enumerate(
                zip(input_seq, output_seq, target_seq, self.cell_state_history)):
            # Display information every 10% of figures.
            if (input_seq.shape[0] > 10) and (i % (input_seq.shape[0] // 10) == 0):
                logger.info("Generating figure {}/{}".format(i, input_seq.shape[0]))

            # Create figure object on the basis of template.
            buf.seek(0)
            fig = pickle.load(buf)
            (ax_memory, ax_attention, ax_snapshot, ax_inputs, ax_targets, ax_predictions) = fig.axes

            # Update displayed values on adequate positions.
            inputs_displayed[:, i] = input_element
            targets_displayed[:, i] = target_element
            predictions_displayed[:, i] = output_elementd

            memory_displayed = memory[0]
            # Get attention of head 0.
            head_attention_displayed[:, i] = wt[0, 0, :]
            snapshot_attention_displayed[:, i] = wt_d[0, 0, :]

            # "Show" data on "axes".
            ax_memory.imshow(np.transpose(memory_displayed), interpolation='nearest', aspect='auto')
            ax_attention.imshow(head_attention_displayed, interpolation='nearest', aspect='auto')
            ax_snapshot.imshow(snapshot_attention_displayed, interpolation='nearest', aspect='auto')
            ax_inputs.imshow(inputs_displayed, interpolation='nearest', aspect='auto')
            ax_targets.imshow(targets_displayed, interpolation='nearest', aspect='auto')
            ax_predictions.imshow(predictions_displayed, interpolation='nearest', aspect='auto')

            # Append figure to a list.
            fig.set_tight_layout(True)
            figs.append(fig)

        # print("--- %s seconds ---" % (time.time() - start_time))
        # Update time plot fir generated list of figures.
        self.plot.update(figs)
        return self.plot.is_closed
