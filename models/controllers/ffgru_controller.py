#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gru_controller.py: pytorch module implementing wrapper for gru controller of NTM."""
__author__ = "Tomasz Kornuta/Ryan L. McAvoy/Younes Bouhadjar"

import torch
from torch import nn
import torch.nn.functional as F
import collections


_GRUStateTuple = collections.namedtuple('GRUStateTuple', ('hidden_state'))
class GRUStateTuple(_GRUStateTuple):
    """Tuple used by gru Cells for storing current/past state information"""
    __slots__ = ()

class FFGRUController(nn.Module):
    def __init__(self, params):
        """ Constructor.
        
        :param params: Dictionary of parameters.
        """

        self.input_size = params["input_size"]
        self.ctrl_hidden_state_size = params["output_size"]
        #self.hidden_state_dim = params["hidden_state_dim"]
        self.num_layers = params["num_layers"]
        self.ff_output_size = params["ff_output_size"]
        assert self.num_layers > 0, "Number of layers should be > 0"

        super(FFGRUController, self).__init__()
      
        self.ff = nn.Linear(self.input_size, self.ff_output_size)
        self.gru = nn.GRUCell(self.ff_output_size, self.ctrl_hidden_state_size)

        
    def init_state(self, batch_size, dtype):
        """
        Returns 'zero' (initial) state tuple.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - object of GRUStateTuple class.
        """
        # Initialize GRU hidden state [BATCH_SIZE x CTRL_HIDDEN_SIZE].
        hidden_state = torch.zeros((batch_size, self.ctrl_hidden_state_size), requires_grad=False).type(dtype)

        return GRUStateTuple(hidden_state)

    def forward(self, x, prev_state_tuple):
        """
        Controller forward function. 
        
        :param x: a Tensor of input data of size [BATCH_SIZE  x INPUT_SIZE] (generally the read data and input word concatenated)
        :param prev_state_tuple: Tuple of the previous hidden and cell state 
        :returns: outputs a Tensor of size  [BATCH_SIZE x OUTPUT_SIZE] and an GRU state tuple.
        """

        hidden_state_prev = prev_state_tuple.hidden_state

        input = self.ff(x)
        hidden_state = self.gru(input, hidden_state_prev)
       
        return hidden_state, GRUStateTuple(hidden_state)


