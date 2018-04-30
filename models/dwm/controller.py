import torch
from torch import nn
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(self, dwm_in_dim, dwm_output_units, dwm_state_units,
                 read_size, update_size):

        """Initialize an Controller.

        :param dwm_in_dim: input size.
        :param dwm_output_units: output size.
        :param dwm_state_units: state size.
        :param read_size: size of data_gen read from memory
        :param update_size: size of data_gen for updating attention and memory
        """
        super(Controller, self).__init__()

        self.read_size = read_size
        self.update_size = update_size

        dwm_ctrl_in_dim = dwm_in_dim + dwm_state_units + self.read_size

        # Output layer
        self.dwm_output_units = dwm_output_units
        if self.dwm_output_units > 0:
            # self.dwm_i2i = nn.Linear(dwm_ctrl_in_dim, dwm_ctrl_in_dim)
            self.dwm_i2o = nn.Linear(dwm_ctrl_in_dim, dwm_output_units)

        # State layer
        self.dwm_i2s = nn.Linear(dwm_ctrl_in_dim, dwm_state_units)

        # Update layer
        self.dwm_i2u = nn.Linear(dwm_ctrl_in_dim, self.update_size)

        #rest parameters
        #self.reset_parameters()

    def forward(self, dwm_input, dwm_state, read_data):
        """
        Calculates the output, the hidden state and the controller parameters
        
        :param dwm_input: Current input (from time t)  [BATCH_SIZE x INPUT_SIZE]
        :param dwm_state: Previous hidden state (from time t-1)  [BATCH_SIZE x dwm_STATE_UNITS]
        :return: Tuple [output, hidden_state, update_data] (update_data contains all of the controller parameters)
        """
        # Concatenate the 3 inputs to controller
        combined = torch.cat((dwm_input, dwm_state, read_data), dim=-1)

        # Get output with activation
        dwm_output = None
        if self.dwm_output_units > 0:
            hidden = combined
            dwm_output = self.dwm_i2o(hidden)
            dwm_output = F.sigmoid(dwm_output)

        # Get the state and update; no activation is applied
        dwm_state = self.dwm_i2s(combined)
        dwm_state = F.sigmoid(dwm_state)

        update_data = self.dwm_i2u(combined)

        return dwm_output, dwm_state, update_data


