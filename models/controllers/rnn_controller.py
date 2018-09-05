#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rnn_controller.py: pytorch module implementing wrapper for RNN controller of NTM."""
__author__ = "Ryan L. McAvoy"


import torch
from torch import nn
import torch.nn.functional as F
import collections
from misc.app_state import AppState

_RNNStateTuple = collections.namedtuple('RNNStateTuple', ('hidden_state'))
class RNNStateTuple(_RNNStateTuple):
    """Tuple used by LSTM Cells for storing current/past state information"""
    __slots__ = ()


class RNNController(nn.Module):
    def __init__(self, params):
        """ Constructor for a RNN
        
        :param params: Dictionary of parameters.
        """

        self.input_size = params["input_size"]
        self.ctrl_hidden_state_size = params["output_size"]
        #self.hidden_state_dim = params["hidden_state_dim"]
        self.non_linearity = params["non_linearity"]
        self.num_layers = params["num_layers"]
        assert self.num_layers > 0, "Number of layers should be > 0"

        super(RNNController, self).__init__()
        full_size=self.input_size+self.ctrl_hidden_state_size
        self.rnn=nn.Linear(full_size, self.ctrl_hidden_state_size)
    
    def init_state(self,  batch_size):
        """
        Returns 'zero' (initial) state tuple.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - object of RNNStateTuple class.
        """
        # Initialize LSTM hidden state [BATCH_SIZE x CTRL_HIDDEN_SIZE].
        dtype = AppState().dtype
        hidden_state = torch.zeros((batch_size, self.ctrl_hidden_state_size), requires_grad=False).type(dtype)

        return RNNStateTuple(hidden_state)        

    def forward(self, inputs, prev_hidden_state_tuple):
        """
        Controller forward function. 
        
        :param inputs: a Tensor of input data of size [BATCH_SIZE  x INPUT_SIZE] (generally the read data and input word concatenated)
        :param prev_state_tuple: Tuple of the previous hidden state 
        :returns: outputs a Tensor of size  [BATCH_SIZE x OUTPUT_SIZE] and an RNN state tuple.
        """
       
        h=prev_hidden_state_tuple[0] 
        combo=torch.cat((inputs,h), dim=-1)
        hidden_state = self.rnn(combo)

        if self.non_linearity == "sigmoid":
            hidden_state=F.sigmoid(hidden_state)
        elif self.non_linearity == "tanh":
            hidden_state=F.tanh(hidden_state)
        elif self.non_linearity == "relu":
            hidden_state=F.relu(hidden_state)
       
        return hidden_state, RNNStateTuple(hidden_state)


