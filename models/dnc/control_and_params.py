import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pdb
import collections
from models.dnc.param_gen import Param_Generator
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'controllers'))
from controller_factory import ControllerFactory

class ControlParams(nn.Module):
    def __init__(self,  output_size, read_size, params):

        """Initialize an Controller.

        :param output_size: output size.
        :param tm_state_units: state size.
        :param read_size: size of data_gen read from memory
        :param num_heads: number of heads (we will have equal numbers of read and write heads)
        """
        super(ControlParams, self).__init__()

        self.read_size = read_size

        # Parse parameters.
        # Set input and hidden  dimensions.
        self.input_size = params["control_bits"] + params["data_bits"]

        tm_ctrl_in_dim = self.input_size + self.read_size

        self.hidden_state_size = params['hidden_state_dim']
        # Get memory parameters.
        self.num_memory_bits = params['memory_content_size']
        
        self.controller_type = params['controller_type']
        self.shift_size = params['shift_size']
        self.num_reads = params['num_reads']
        self.num_writes = params['num_writes']
        self.non_linearity = params['non_linearity']


        # State layer
        controller_params = {
           "name": self.controller_type,
           "input_size": tm_ctrl_in_dim,
           "output_size": self.hidden_state_size,
           "num_layers": 1,
           "non_linearity" : self.non_linearity
        }

        self.tm_i2s = ControllerFactory.build_model(controller_params)

        
        self.tm_i2o = nn.Linear(self.hidden_state_size, output_size)

        # Update layer
        self.tm_i2u = Param_Generator(self.hidden_state_size, word_size=self.num_memory_bits,num_reads=self.num_reads,num_writes=self.num_writes,shift_size=self.shift_size)
    
    def init_state(self,  batch_size):
        """
        Returns 'zero' (initial) state tuple.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - object of LSTMStateTuple class.
        """
        return self.tm_i2s.init_state(batch_size)

    def forward(self, tm_input, prev_ctrl_state_tuple, read_data):
        """
        Calculates the output, the hidden state and the controller parameters
        
        :param tm_input: Current input (from time t)  [BATCH_SIZE x INPUT_SIZE]
        :param tm_state: Previous hidden state (from time t-1)  [BATCH_SIZE x TM_STATE_UNITS]
        :return: Tuple [output, hidden_state, update_data] (update_data contains all of the controller parameters)
        """
        # Concatenate the 2 inputs to controller
        combined = torch.cat((tm_input, read_data), dim=-1)

        tm_state, ctrl_state_tuple = self.tm_i2s(combined, prev_ctrl_state_tuple)

        tm_output = self.tm_i2o(tm_state)

        update_data = self.tm_i2u(tm_state)

        return tm_output, ctrl_state_tuple, update_data


