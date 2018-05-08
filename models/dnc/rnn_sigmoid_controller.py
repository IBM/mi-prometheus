import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import collections

_RNNStateTuple = collections.namedtuple('RNNStateTuple', ('hidden_state'))
class RNNStateTuple(_RNNStateTuple):
    """Tuple used by LSTM Cells for storing current/past state information"""
    __slots__ = ()


class RNNController(nn.Module):
    def __init__(self, params):
        self.input_size = params["input_size"]
        self.ctrl_hidden_state_size = params["output_size"]
        #self.hidden_state_dim = params["hidden_state_dim"]
        self.num_layers = params["num_layers"]
        assert self.num_layers > 0, "Number of layers should be > 0"

        super(RNNController, self).__init__()
        full_size=self.input_size+self.ctrl_hidden_state_size
        self.rnn=nn.Linear(full_size, self.ctrl_hidden_state_size)
    
    def init_state(self,  batch_size):
        """
        Returns 'zero' (initial) state tuple.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - object of LSTMStateTuple class.
        """
        # Initialize LSTM hidden state [BATCH_SIZE x CTRL_HIDDEN_SIZE].
        hidden_state = torch.zeros(batch_size, self.ctrl_hidden_state_size, requires_grad=False)

        return RNNStateTuple(hidden_state)        

    def forward(self, inputs, prev_hidden_state_tuple):
       
        h=prev_hidden_state_tuple[0] 
        combo=torch.cat((inputs,h), dim=-1)
        #print(combo.shape)
        hidden_state = F.sigmoid(self.rnn(combo))
       
        return hidden_state, RNNStateTuple(hidden_state)


