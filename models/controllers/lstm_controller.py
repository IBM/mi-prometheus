#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lstm_controller.py: pytorch module implementing wrapper for lstm controller of NTM."""
__author__ = "Tomasz Kornuta/Ryan L. McAvoy"

import torch
from torch import nn
import collections
from misc.app_state import AppState

_LSTMStateTuple = collections.namedtuple(
    'LSTMStateTuple', ('hidden_state', 'cell_state'))


class LSTMStateTuple(_LSTMStateTuple):
    """
    Tuple used by LSTM Cells for storing current/past state information.
    """
    __slots__ = ()


class LSTMController(nn.Module):
    def __init__(self, params):
        """
        Constructor.

        :param params: Dictionary of parameters.

        """

        self.input_size = params["input_size"]
        self.ctrl_hidden_state_size = params["output_size"]
        #self.hidden_state_dim = params["hidden_state_dim"]
        self.num_layers = params["num_layers"]
        assert self.num_layers > 0, "Number of layers should be > 0"

        super(LSTMController, self).__init__()

        self.lstm = nn.LSTMCell(self.input_size, self.ctrl_hidden_state_size)

    def init_state(self, batch_size):
        """
        Returns 'zero' (initial) state tuple.

        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - object of LSTMStateTuple class.

        """
        dtype = AppState().dtype
        # Initialize LSTM hidden state [BATCH_SIZE x CTRL_HIDDEN_SIZE].
        hidden_state = torch.zeros(
            (batch_size,
             self.ctrl_hidden_state_size),
            requires_grad=False).type(dtype)
        # Initialize LSTM memory cell [BATCH_SIZE x CTRL_HIDDEN_SIZE].
        cell_state = torch.zeros(
            (batch_size,
             self.ctrl_hidden_state_size),
            requires_grad=False).type(dtype)

        return LSTMStateTuple(hidden_state, cell_state)

    def forward(self, x, prev_state_tuple):
        """
        Controller forward function.

        :param x: a Tensor of input data of size [BATCH_SIZE  x INPUT_SIZE] (generally the read data and input word concatenated)
        :param prev_state_tuple: Tuple of the previous hidden and cell state
        :returns: outputs a Tensor of size  [BATCH_SIZE x OUTPUT_SIZE] and an LSTM state tuple.

        """

        hidden_state, cell_state = self.lstm(x, prev_state_tuple)

        return hidden_state, LSTMStateTuple(hidden_state, cell_state)
