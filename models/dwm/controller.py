import torch
from torch import nn
import torch.nn.functional as F


class Controller(nn.Module):
    def __init__(self, in_dim, output_units, state_units,
                 read_size, update_size):

        """Initialize an Controller.

        :param in_dim: input size.
        :param output_units: output size.
        :param state_units: state size.
        :param read_size: size of data_gen read from memory
        :param update_size: size of data_gen for updating attention and memory
        """
        super(Controller, self).__init__()

        self.read_size = read_size
        self.update_size = update_size

        ctrl_in_dim = in_dim + state_units + self.read_size

        # Output layer
        self.output_units = output_units
        if self.output_units > 0:
            # self.i2i = nn.Linear(ctrl_in_dim, ctrl_in_dim)
            self.i2o = nn.Linear(ctrl_in_dim, output_units)

        # State layer
        self.i2s = nn.Linear(ctrl_in_dim, state_units)

        # Update layer
        self.i2u = nn.Linear(ctrl_in_dim, self.update_size)

        #rest parameters
        #self.reset_parameters()

    def forward(self, input, wt_head_prev, read_data):
        """
        Calculates the output, the hidden state and the controller parameters
        
        :param input: Current input (from time t)  [BATCH_SIZE x INPUT_SIZE]
        :param state: Previous hidden state (from time t-1)  [BATCH_SIZE x STATE_UNITS]
        :return: Tuple [output, hidden_state, update_data] (update_data contains all of the controller parameters)
        """
        # Concatenate the 3 inputs to controller
        combined = torch.cat((input, wt_head_prev, read_data), dim=-1)

        # Get the state and update; no activation is applied
        state = self.i2s(combined)
        state = F.sigmoid(state)

        # Get output with activation
        output = self.i2o(combined)
        output = F.sigmoid(output)

        # update attentional parameters and memory update parameters
        update_data = self.i2u(combined)

        return output, state, update_data


