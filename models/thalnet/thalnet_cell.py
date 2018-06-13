import torch
from torch import nn
import collections
from models.thalnet.module import Module


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
        self.modules_thalnet = nn.ModuleList()

        self.modules_thalnet.append(Module(center_size=self.center_size,
                            context_size=self.context_input_size,
                            center_size_per_module=self.center_size_per_module,
                            input_size=self.input_size,
                            output_size=0))

        self.modules_thalnet.extend([Module(center_size=self.center_size,
                            context_size=self.context_input_size,
                            center_size_per_module=self.center_size_per_module,
                            input_size=0,
                            output_size=self.output_size if i == self.num_modules - 1 else 0)
                                 for i in range(1, self.num_modules)])

    def init_state(self, batch_size, dtype):

        # module and center state initialisation
        states = [self.modules_thalnet[i].init_state(batch_size, dtype) for i in range(self.num_modules)]

        return states

    def forward(self, inputs, prev_state):
        prev_center_states = [prev_state[i][0] for i in range(self.num_modules)]
        prev_module_states = [prev_state[i][1] for i in range(self.num_modules)]

        # Concatenate all the centers
        prev_center_states = torch.cat(prev_center_states, dim=1)

        states = []
        # run the different modules, they share all the same center
        for module, prev_module_state in zip(self.modules_thalnet, prev_module_states):
            output, center_feature, module_state = module(inputs, prev_center_states, prev_module_state)
            states.append((center_feature, module_state))

        return output, states
