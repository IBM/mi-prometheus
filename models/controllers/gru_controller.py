#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gru_controller.py: pytorch module implementing wrapper for gru controller of NTM."""
__author__ = "Tomasz Kornuta/Ryan L. McAvoy/Younes Bouhadjar"

import torch
from torch import nn
import collections
from misc.app_state import AppState

_GRUStateTuple = collections.namedtuple('GRUStateTuple', ('hidden_state'))


class GRUStateTuple(_GRUStateTuple):
    """
    Tuple used by GRU Cells for storing current/past state information.
    """
    __slots__ = ()


class GRUController(nn.Module):
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

        super(GRUController, self).__init__()

        self.gru = nn.GRUCell(self.input_size, self.ctrl_hidden_state_size)

    def init_state(self, batch_size):
        """
        Returns 'zero' (initial) state tuple.

        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - object of GRUStateTuple class.
        :returns: Initial state tuple - object of GRUStateTuple class.

        """
        # Initialize GRU hidden state [BATCH_SIZE x CTRL_HIDDEN_SIZE].
        dtype = AppState().dtype
        hidden_state = torch.zeros(
            (batch_size,
             self.ctrl_hidden_state_size),
            requires_grad=False).type(dtype)

        return GRUStateTuple(hidden_state)

    def forward(self, x, prev_state_tuple):
        """
        Controller forward function.

        :param x: a Tensor of input data of size [BATCH_SIZE  x INPUT_SIZE] (generally the read data and input word concatenated)
        :param prev_state_tuple: Tuple of the previous hidden and cell state
        :returns: outputs a Tensor of size  [BATCH_SIZE x OUTPUT_SIZE] and an GRU state tuple.

        """

        hidden_state_prev = prev_state_tuple.hidden_state
        hidden_state = self.gru(x, hidden_state_prev)

        return hidden_state, GRUStateTuple(hidden_state)
