import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import collections


_LSTMStateTuple = collections.namedtuple('NTMStateTuple', ('hidden_state', 'cell_state'))
class LSTMStateTuple(_LSTMStateTuple):
    """Tuple used by LSTM Cells for storing current/past state information"""
    __slots__ = ()

class LSTMController(nn.Module):
    def __init__(self, params, plot_active=False):
        self.input_size = params["input_size"]
        self.ctrl_hidden_state_size = params["output_size"]
        #self.hidden_state_dim = params["hidden_state_dim"]
        self.num_layers = params["num_layers"]
        assert self.num_layers > 0, "Number of layers should be > 0"

        super(LSTMController, self).__init__()
        #print(params)
#        tm_size=self.tm_in_dim+self.hidden_state_dim
       
        self.lstm=nn.LSTMCell(self.input_size, self.ctrl_hidden_state_size)
        
    def init_state(self,  batch_size):
        """
        Returns 'zero' (initial) state tuple.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - object of LSTMStateTuple class.
        """
        # Initialize LSTM hidden state [BATCH_SIZE x CTRL_HIDDEN_SIZE].
        hidden_state = torch.zeros(batch_size, self.ctrl_hidden_state_size, requires_grad=False)
        # Initialize LSTM memory cell [BATCH_SIZE x CTRL_HIDDEN_SIZE].
        cell_state = torch.zeros(batch_size, self.ctrl_hidden_state_size, requires_grad=False)
        
        return LSTMStateTuple(hidden_state,  cell_state)


    def forward(self, x, prev_state_tuple):
        #if not hasattr(self, "c0"):
        #   # Create the internal state tensors
        #    self.c0 = torch.zeros((h.shape), requires_grad=False)
        
        hidden_state, cell_state = self.lstm(x,prev_state_tuple)
       
        return hidden_state, LSTMStateTuple(hidden_state,  cell_state)


