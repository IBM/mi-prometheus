import torch
from torch import nn
import collections
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'controllers'))
from controller_factory import ControllerFactory

# Helper collection type.
_ThalNetModuleState = collections.namedtuple('ThalNetModuleState', ('module'))

class ThalNetModuleState(_ThalNetModuleState):
    """Tuple used by interface for storing current/past state information"""

class Module(nn.Module):
    def __init__(self,
                 center_size,
                 context_size,
                 center_size_per_module,
                 input_size,
                 output_size):
        super(Module, self).__init__()

        self.center_size = center_size
        self.context_size = context_size
        self.center_size_per_module = center_size_per_module

        self.output_size = output_size
        self.input_size = input_size

        # Reading mechanism
        self.fc_context = nn.utils.weight_norm(nn.Linear(self.center_size, self.context_size), name='weight')

        # FeedForward & GRU
        self.input_context_size = self.input_size + self.context_size
        self.gru_hidden_size = self.output_size + self.center_size_per_module
        self.controller_type = 'rnn'
        self.non_linearity = ''

        # Set module
        controller_params = {
            "name": self.controller_type,
            "input_size": self.input_context_size,
            "output_size": self.gru_hidden_size,
            "num_layers": 1,
            "non_linearity": self.non_linearity
        }

        self.module_cell = ControllerFactory.build_model(controller_params)

    def init_state(self, batch_size, dtype):
        # module state initialisation
        tuple_module_states = self.module_cell.init_state(batch_size, dtype)

        # center state initialisation
        center_state_per_module = torch.randn((batch_size, self.center_size_per_module))

        return center_state_per_module, tuple_module_states

    def forward(self, inputs, prev_center_state, prev_tuple_module_state):
        """
        :return: output, new_center_features, new_module_state
        """
        if inputs is not None:
            if len(inputs.size()) <= 1 or len(inputs.size()) >= 4:
                print('check inputs size of thalnet cell')
                exit(-1)

            if len(inputs.size()) == 3:
                # inputs_size : [batch_size, num_channel, input_size]
                # select channel
                inputs = inputs[:, 0, :]

        # get the context_input and the inputs of the module
        context_input = self.fc_context(prev_center_state)
        inputs = torch.cat((inputs, context_input), dim=1) if self.input_size else context_input

        # apply FeedForward & GRU
        tuple_gru_output = self.module_cell(inputs, prev_tuple_module_state)
        tuple_module_state = tuple_gru_output

        output, center_feature_output = torch.split(tuple_gru_output[0],
                                        [self.output_size, self.center_size_per_module], dim=1) if self.output_size else (None, tuple_gru_output[0])

        return output, center_feature_output, tuple_module_state