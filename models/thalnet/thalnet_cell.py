import torch
from torch import nn
import collections

# Helper collection type.
_ThalNetModuleState = collections.namedtuple('ThalNetModuleState', ('module'))

class ThalNetModuleState(_ThalNetModuleState):
    """Tuple used by interface for storing current/past state information"""

class FfGruModule(nn.Module):
    def __init__(self,
                 center_size,
                 context_input_size,
                 center_output_size,
                 input_size,
                 output_size):
        super(FfGruModule, self).__init__()

        self.center_size = center_size
        self.context_input_size = context_input_size
        self.center_output_size = center_output_size

        self.output_size = output_size
        self.input_size = input_size

        # Reading mechanism
        self.fc_context = nn.utils.weight_norm(nn.Linear(self.center_size, self.context_input_size), name='weight')

        # FeedForward & GRU
        self.input_context_size = self.input_size + self.context_input_size
        self.gru_hidden_size = self.output_size + self.center_output_size

        self.fc = nn.Linear(self.input_context_size, self.center_output_size)
        self.grucell = nn.GRUCell(self.center_output_size, self.gru_hidden_size)

    def forward(self, inputs, center_state, tuple_module_state):
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
        context_input = self.fc_context(center_state)
        inputs = torch.cat((inputs, context_input), dim=1) if self.input_size else context_input

        # apply FeedForward & GRU
        inputs = self.fc(inputs)
        gru_output = self.grucell(inputs, tuple_module_state)
        new_module_state = gru_output

        output, center_feature_output = torch.split(gru_output,
                                        [self.output_size, self.center_output_size], dim=1) if self.output_size else (None, gru_output)

        return output, center_feature_output, ThalNetModuleState(new_module_state)


class ThalNetCell(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 context_input_size,
                 center_size_per_module,
                 num_modules):

        self.context_input_size = context_input_size
        self.input_size = input_size
        self.output_size = output_size
        self.center_size = num_modules * center_size_per_module
        self.center_size_per_module = center_size_per_module
        self.num_modules = num_modules
        super(ThalNetCell, self).__init__()

        # init GRU-center cell
        self.modules_gru = nn.ModuleList()

        self.modules_gru.append(FfGruModule(center_size=self.center_size,
                            context_input_size=self.context_input_size,
                            center_output_size=self.center_size_per_module,
                            input_size=self.input_size,
                            output_size=0))

        self.modules_gru.extend([FfGruModule(center_size=self.center_size,
                            context_input_size=self.context_input_size,
                            center_output_size=self.center_size_per_module,
                            input_size=0,
                            output_size=self.output_size if i == self.num_modules - 1 else 0)
                                 for i in range(1, self.num_modules)])

    def forward(self, inputs, state_prev):
        center_state_per_module_prev = state_prev[:self.num_modules]
        module_states_prev = state_prev[self.num_modules:]

        center_state = torch.cat(center_state_per_module_prev, dim=1)

        module_states_next = []
        center_features_next = []
        # run the different modules, they share all the same center
        for module, module_state_prev in zip(self.modules_gru, module_states_prev):
            output, center_features, module_state = module(inputs, center_state, module_state_prev)
            center_features_next.append(center_features)
            module_states_next.append(module_state)

        return output, center_features_next + module_states_next
