import torch
from torch import nn
from torch.autograd import Variable

from matplotlib.figure import Figure
from matplotlib import ticker
import numpy as np

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
        self.plot_active = params["plot_memory"]
        self.label = params["name"]
        self.app_state = AppState()

        # This is for the time plot
        self.cell_state_history = None

        super(DWM, self).__init__()

        # Create the DWM components
        self.DWMCell = DWMCell(self.in_dim, self.output_units, self.state_units,
                               self.num_heads, self.is_cam, self.num_shift, self.M)

    def forward(self, inputs):       # x : batch_size, seq_len, input_size
        """
        Runs the DWM cell and plots if necessary
        
        :param x: input sequence  [BATCH_SIZE x seq_len x input_size ]
        :param state: Input hidden state  [BATCH_SIZE x state_size]
        :return: Tuple [output, hidden_state]
        """
        if self.app_state.visualize:
            self.cell_state_history = []

        output = None
        batch_size = inputs.size(0)
        seq_length = inputs.size(1)
        memory_addresses_size = self.memory_addresses_size
        if memory_addresses_size == -1:
            memory_addresses_size = seq_length  # a hack for now

        # init state
        cell_state = self.init_state(memory_addresses_size, batch_size)
        for j in range(seq_length):
            output_cell, cell_state = self.DWMCell(inputs[..., j, :], cell_state)

            if output_cell is None:
                continue

            output_cell = output_cell[..., None, :]
            if output is None:
                output = output_cell
                continue

            # concatenate output
            output = torch.cat([output, output_cell], dim=-2)

            # This is for the time plot
            if self.app_state.visualize:
                self.cell_state_history.append((cell_state[3].detach().numpy(),
                                                cell_state[1].detach().numpy()))

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

    # Overrides the one in ModelBase
    def plot_sequence(self, input_seq, output_seq, target_seq):
        figs = []

        input_seq = input_seq.numpy()
        output_seq = output_seq.numpy()
        target_seq = target_seq.numpy()

        pred_matrix = np.zeros(output_seq.shape)

        for i, (input_word, output_word, target_word, (memory, wt)) \
                in enumerate(zip(input_seq,
                                 output_seq,
                                 target_seq,
                                 self.cell_state_history)):

            pred_matrix[i] = output_word

            fig = Figure()
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(212)
            ax3 = fig.add_subplot(222)

            ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax1.set_title("Attention", fontname='Times New Roman', fontsize=15)
            ax1.plot(np.arange(wt.shape[-1]), wt[0, 0, :], 'go')

            ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax2.set_ylabel("Word size", fontname='Times New Roman', fontsize=15)
            ax2.set_xlabel("Memory addresses", fontname='Times New Roman', fontsize=15)
            ax2.set_title("Task: xxx", fontname='Times New Roman', fontsize=15)

            ax2.imshow(memory[0, :, :], interpolation='nearest')

            ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax3.set_title("Prediction", fontname='Times New Roman', fontsize=15)
            ax3.imshow(np.transpose(pred_matrix, [1, 0]))

            figs.append(fig)

        self.plot.update(figs)
        return self.plot.is_closed
