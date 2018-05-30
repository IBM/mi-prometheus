import torch
from torch import nn
import torch.nn.functional as F

class FfGruModule:
    def __init__(self,
                 center_size,
                 context_input_size,
                 center_output_size,
                 input_size,
                 output_size,
                 name):
        self.name = name

        self.center_size = center_size
        self.context_input_size = context_input_size
        self.center_output_size = center_output_size

        self.input_size = input_size
        self.output_size = output_size


        # GRU + FeedForward
        self.fc = nn.Linear(input_size, center_output_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers)


        self.num_gru_units = self.output_size + self.center_output_size

    def __call__(self, inputs, center_state, module_state):
        """
        :return: output, new_center_features, new_module_state
        """
        reading_weights = tf.get_variable('reading_weights',shape=[self.center_size,self.context_input_size],initializer=tf.truncated_normal_initializer(stddev=0.1))

        context_input = torch.matmul(center_state, torch.clip_by_norm(reading_weights,1.0))

        inputs = torch.cat([inputs, context_input], axis=1) if self.input_size else context_input

        inputs = self.fc(inputs)

        gru = tf.nn.rnn_cell.GRUCell(self.num_gru_units)

        gru_output, new_module_state = self.gru(inputs, module_state)

        output, center_feature_output = tf.split(gru_output,
                                        [self.output_size, self.center_output_size], axis=1) if self.output_size else (None, gru_output)

        return output, center_feature_output, new_module_state


class ThalNetCell(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 context_input_size,
                 center_size_per_module,
                 num_modules,
                 hidden_size):

        self.context_input_size = context_input_size
        self.input_size = input_size
        self.output_size = output_size
        self.center_size = num_modules * center_size_per_module
        self.enter_size_per_module = center_size_per_module
        self.num_modules = num_modules
        super(ThalNetCell, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, )

    def state_size(self):
        return [module.center_output_size for module in self.modules] + \
               [module.num_gru_units for module in self.modules]

    def output_size(self):
        return self._output_size

    @lazy_property
    def modules(self):
        return [FfGruModule(center_size=self._center_size,
                            context_input_size=self._context_input_size,
                            center_output_size=self.center_size_per_module,
                            input_size=self._input_size if i == 0 else 0,
                            output_size=self.output_size if i == self._num_modules - 1 else 0,
                            name=f'module{i}') for i in range(self._num_modules)]

    def forward(self, inputs, state, scope=None):
        center_state_per_module = state[:self._num_modules]
        module_states = state[self._num_modules:]

        center_state = torch.cat(center_state_per_module, axis=1)

        outputs, new_center_features, new_module_states = unzip(
            [module(inputs if module.input_size else None, center_state=center_state, module_state=module_state)
             for module, module_state in zip(self.modules, module_states)])

        output = single([o for o in outputs if o is not None])

        return output, list((new_center_features + new_module_states))
