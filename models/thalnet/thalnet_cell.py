import torch
from torch import nn
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F
from models.thalnet.utils import single, unzip


class FfGruModule(nn.Module):
    def __init__(self,
                 center_size,
                 context_input_size,
                 center_output_size,
                 input_size,
                 output_size,
                 name):
        self.name = name
        super(FfGruModule, self).__init__()

        self.center_size = center_size
        self.context_input_size = context_input_size
        self.center_output_size = center_output_size

        self.output_size = output_size
        self.input_size = input_size

        # reading mechanism
        self.fc_context = nn.utils.weight_norm(nn.Linear(self.center_size, self.context_input_size), name='weight')

        # FeedForward & GRU
        self.input_context_size = self.input_size + self.context_input_size
        self.gru_hidden_size = self.output_size + self.center_output_size

        self.fc = nn.Linear(self.input_context_size, self.center_output_size)
        self.grucell = nn.GRUCell(self.center_output_size, self.gru_hidden_size)

        self.fo = nn.Linear(self.gru_hidden_size, self.center_output_size)

    def forward(self, inputs, center_state, module_state):
        """
        :return: output, new_center_features, new_module_state
        """
        if inputs is not None:
            print(inputs.size())
            inputs = inputs[:, 0, :]

        # get the context_input and the inputs of the module
        context_input = self.fc_context(center_state)
        inputs = torch.cat((inputs, context_input), dim=1) if self.input_size else context_input

        # apply FeedForward & GRU
        inputs = self.fc(inputs)
        gru_output = self.grucell(inputs, module_state)
        new_module_state = gru_output

        output, center_feature_output = torch.split(gru_output,
                                        [self.output_size, self.center_output_size], dim=1) if self.output_size else (None, gru_output)

        return output, center_feature_output, new_module_state


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
                            input_size = 0,
                            output_size= 0,
                            name=f'module{0}'))

        self.modules_gru.extend([FfGruModule(center_size=self.center_size,
                            context_input_size=self.context_input_size,
                            center_output_size=self.center_size_per_module,
                            input_size=self.input_size,
                            output_size=self.output_size if i == self.num_modules - 1 else 0,
                            name=f'module{i}') for i in range(1, self.num_modules)])

    def forward(self, inputs, state, scope=None):
        center_state_per_module = state[:self.num_modules]
        module_states = state[self.num_modules:]

        center_state = torch.cat(center_state_per_module, dim=1)

        outputs, new_center_features, new_module_states = unzip(
            [module(inputs if module.input_size else None, center_state=center_state, module_state=module_state)
             for module, module_state in zip(self.modules_gru, module_states)])

        output = single([o for o in outputs if o is not None])

        return output, list((new_center_features + new_module_states))
