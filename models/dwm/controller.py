import torch
from torch import nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'controllers'))
from controller_factory import ControllerFactory


class Controller(nn.Module):
    def __init__(self, in_dim, output_units, state_units,
                 read_size, update_size):

        """Initialize the Controller.

        :param in_dim: input size.
        :param output_units: output size.
        :param state_units: state size.
        :param read_size: size of data_gen read from memory
        :param update_size: total number of parameters for updating attention and memory
        """
        super(Controller, self).__init__()

        self.read_size = read_size
        self.update_size = update_size
        self.state_units = state_units

        self.ctrl_in_dim = in_dim + self.read_size
        self.ctrl_in_state_dim = in_dim + state_units + self.read_size

        # Output layer
        self.output_units = output_units

        # State layer dictionary
        self.controller_type = 'rnn'
        self.non_linearity = 'sigmoid'

        controller_params = {
            "name": self.controller_type,
            "input_size": self.ctrl_in_dim,
            "output_size": self.state_units,
            "num_layers": 1,
            "non_linearity": self.non_linearity
        }

        # State layer
        self.i2s = ControllerFactory.build_model(controller_params)

        # Update layer
        self.i2u = nn.Linear(self.ctrl_in_state_dim, self.update_size)

        # Output layer
        self.i2o = nn.Linear(self.ctrl_in_state_dim, self.output_units)

    def init_state(self, batch_size, dtype):
        """
        Returns 'zero' (initial) state tuple.

        :param batch_size: Size of the batch in given iteraction/epoch.
        :param dtype
        :returns: Initial state tuple - object of LSTMStateTuple class.
        """

        return self.i2s.init_state(batch_size, dtype)

    def forward(self, input, tuple_state_prev, read_data):
        """
        Calculates the output, the hidden state and the controller parameters
        
        :param input of shape (batch_size, in_dim): Current input (from time t)
        :param tuple_state_prev: (hidden_state) object of class RNNStateTuple.
        hidden_state of shape (batch_size, state_units): Previous hidden state (from time t-1)
        :param read_data of shape (batch_size, read_size): read data from memory (from time t)

        :return: output of shape (batch_size, output_units)
                 tuple_state: (new_hidden_state)
                 update_data of shape (batch_size, update_size): contains all of the controller parameters
        """
        # Concatenate the 3 inputs to controller
        combined = torch.cat((input, read_data), dim=-1)
        combined_with_state = torch.cat((combined, tuple_state_prev[0]), dim=-1)

        # Get the state and update; no activation is applied
        state, tuple_state = self.i2s(combined, tuple_state_prev)

        # Get output with activation
        output = self.i2o(combined_with_state)

        # update attentional parameters and memory update parameters
        update_data = self.i2u(combined_with_state)

        return output, tuple_state, update_data


