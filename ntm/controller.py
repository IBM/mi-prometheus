import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pdb


class Controller(nn.Module):
    def __init__(self, tm_in_dim, tm_output_units, tm_state_units,
                 read_size, update_size):

        """Initialize an Controller.

        :param tm_in_dim: input size.
        :param tm_output_units: output size.
        :param tm_state_units: state size.
        :param read_size: size of data_gen read from memory
        :param update_size: size of data_gen for updating attention and memory
        """
        super(Controller, self).__init__()

        self.read_size = read_size
        self.update_size = update_size

        tm_ctrl_in_dim = tm_in_dim + tm_state_units + self.read_size

        # Output layer
        self.tm_output_units = tm_output_units
        if self.tm_output_units > 0:
            # self.tm_i2i = nn.Linear(tm_ctrl_in_dim, tm_ctrl_in_dim)
            self.tm_i2o = nn.Linear(tm_ctrl_in_dim, tm_output_units)

        # State layer
        self.tm_i2s = nn.Linear(tm_ctrl_in_dim, tm_state_units)

        # Update layer
        self.tm_i2u = nn.Linear(tm_ctrl_in_dim, self.update_size)

        #rest parameters
        #self.reset_parameters()

    def forward(self, tm_input, tm_state, read_data):
        # Concatenate the 3 inputs to controller
        combined = torch.cat((tm_input, tm_state, read_data), dim=-1)

        # Get output with activation
        tm_output = None
        if self.tm_output_units > 0:
            hidden = combined
            tm_output = self.tm_i2o(hidden)
            tm_output = F.sigmoid(tm_output)

        # Get the state and update; no activation is applied
        tm_state = self.tm_i2s(combined)
        tm_state = F.sigmoid(tm_state)

        update_data = self.tm_i2u(combined)

        return tm_output, tm_state, update_data

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform(self.tm_i2s.weight, gain=1.4)
        nn.init.normal(self.tm_i2s.bias, std=0.01)

        nn.init.xavier_uniform(self.tm_i2o.weight, gain=1.4)
        nn.init.normal(self.tm_i2o.bias, std=0.01)

        nn.init.xavier_uniform(self.tm_i2u.weight, gain=1.4)
        nn.init.normal(self.tm_i2u.bias, std=0.01)
