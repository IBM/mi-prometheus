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

#from models.controllers.controller_factory import ControllerFactory

class Controller(nn.Module):
    def __init__(self, tm_in_dim, tm_output_units, tm_state_units,
                 read_size, params):

        """Initialize an Controller.

        :param tm_in_dim: input size.
        :param tm_output_units: output size.
        :param tm_state_units: state size.
        :param read_size: size of data_gen read from memory
        :param num_heads: number of heads (we will have equal numbers of read and write heads)
        """
        super(Controller, self).__init__()

        self.read_size = read_size

        #tm_ctrl_in_dim = tm_in_dim + tm_state_units + self.read_size
        tm_ctrl_in_dim = tm_in_dim + self.read_size

        # Output eayer
        self.tm_output_units = tm_output_units
        #if self.tm_output_units > 0:
            # self.tm_i2i = nn.Linear(tm_ctrl_in_dim, tm_ctrl_in_dim)

        # Parse parameters.
        # Set input and hidden  dimensions.
        self.input_size = params["control_bits"] + params["data_bits"]
        self.ctrl_hidden_state_size = params['hidden_state_dim']
        # Get memory parameters.
        self.num_memory_bits = params['memory_content_size']
        # TODO - move memory size somewhere?
        #self.num_memory_addresses = params['memory_addresses_size']
        self.controller_type = params['controller_type']
        self.shift_size = params['shift_size']
        self.num_reads = params['num_reads']
        self.num_writes = params['num_writes']
        self.non_linearity = params['non_linearity']


        # State layer
        #self.tm_i2s = nn.LSTMCell(tm_ctrl_in_dim,tm_state_units)
        controller_params = {
           "name": self.controller_type,
           "input_size": tm_ctrl_in_dim,
           "output_size": self.ctrl_hidden_state_size,
           "num_layers": 1,
           "non_linearity" : self.non_linearity
        }
   
        #self.tm_i2s = nn.Linear(tm_ctrl_in_dim, tm_state_units)
        self.tm_i2s = ControllerFactory.build_model(controller_params)

        
        self.tm_i2o = nn.Linear(tm_state_units, tm_output_units)

        # Update layer
        self.tm_i2u = Param_Generator(tm_state_units, word_size=self.num_memory_bits,num_reads=self.num_reads,num_writes=self.num_writes,shift_size=self.shift_size)
        #self.tm_i2u = Param_Generator(tm_ctrl_in_dim, word_size=self.read_size)
        #self.tm_i2u = Param_Generator(tm_state_units, word_size=self.read_size)
        #self.tm_i2u = nn.Linear(tm_ctrl_in_dim, self.update_size)

        #rest parameters
        #self.reset_parameters()
    def init_state(self,  batch_size, dtype):
        """
        Returns 'zero' (initial) state tuple.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :returns: Initial state tuple - object of LSTMStateTuple class.
        """
        # Initialize LSTM hidden state [BATCH_SIZE x CTRL_HIDDEN_SIZE].
        #hidden_state = torch.zeros(batch_size, self.ctrl_hidden_state_size, requires_grad=False)
        # Initialize LSTM memory cell [BATCH_SIZE x CTRL_HIDDEN_SIZE].
       # cell_state = torch.zeros(batch_size, self.ctrl_hidden_state_size, requires_grad=False)

        return self.tm_i2s.init_state(batch_size, dtype)

    def forward(self, tm_input, prev_ctrl_state_tuple, read_data):
        """
        Calculates the output, the hidden state and the controller parameters
        
        :param tm_input: Current input (from time t)  [BATCH_SIZE x INPUT_SIZE]
        :param tm_state: Previous hidden state (from time t-1)  [BATCH_SIZE x TM_STATE_UNITS]
        :return: Tuple [output, hidden_state, update_data] (update_data contains all of the controller parameters)
        """
        # Concatenate the 3 inputs to controller
        combined = torch.cat((tm_input, read_data), dim=-1)

        tm_state, ctrl_state_tuple = self.tm_i2s(combined,prev_ctrl_state_tuple)

        tm_output = self.tm_i2o(tm_state)

        update_data = self.tm_i2u(tm_state)

        return tm_output, ctrl_state_tuple, update_data


class RNN(nn.Module):
    def __init__(self, params, plot_active=False):
        self.tm_in_dim = params["input_size"]
        self.hidden_state_dim = params["output_size"]
        #self.hidden_state_dim = params["hidden_state_dim"]
        self.num_layers = params["num_layers"]
        assert self.num_layers > 0, "Number of layers should be > 0"

        super(RNN, self).__init__()
        print(params)
        tm_size=self.tm_in_dim+self.hidden_state_dim
        self.layers=nn.RNNCell(self.tm_in_dim, self.hidden_state_dim)
        

    def forward(self, x, h, prev_state_tuple):
        
        tm_state = self.layers(x,h)
       
        return tm_state, ()

    
class DeepLSTMCell(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_state_dim,num_layers=1):
        self.tm_in_dim = input_dim
        self.data_bits = output_dim
        self.hidden_state_dim = hidden_state_dim
        self.num_layers = num_layers
        assert self.num_layers > 0, "Number of LSTM layers should be > 0"

        super(LSTM, self).__init__()

        # Create the LSTM layers
        self.lstm_layers = nn.ModuleList()
        sizes=[self.tm_in_dim]
        sizes=sizes+[self.hidden_state_dim for _ in range(1,self.num_layers)]
        sizes=sizes+[self.data_bits]

        self.lstm_layers.append(nn.LSTMCell(sizes[0], sizes[1]))
        self.lstm_layers.extend([nn.LSTMCell(sizes[i], sizes[i+1])
                                 for i in range(1, self.num_layers)])

        self.linear = nn.Linear(self.hidden_state_dim, self.data_bits)

    def forward(self, x):
        # Create the hidden state tensors
        h = [Variable(torch.zeros(x.size(0), self.hidden_state_dim).type(dtype), requires_grad=False)
                  for _ in range(self.num_layers)]

        # Create the internal state tensors
        c = [Variable(torch.zeros(x.size(0), self.hidden_state_dim).type(dtype), requires_grad=False)
                  for _ in range(self.num_layers)]
        outputs = []

        for x_t in x.chunk(x.size(1), dim=1):
            h[0], c[0] = self.lstm_layers[0](x_t.squeeze(1), (h[0], c[0]))
            for i in range(1, self.num_layers):
                h[i], c[i] = self.lstm_layers[i](h[i-1], (h[i], c[i]))

            #out = self.linear(h[-1])
            outputs += [h[-1]]

        outputs = F.sigmoid(torch.stack(outputs, 1))
        return outputs
