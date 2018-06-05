import torch

from torch import nn
from models.thalnet.thalnet_cell import ThalNetCell


class THALNET(nn.Module):

    def __init__(self, params):
        """Initialize an THALNET Layer. """
        self.context_input_size = params['context_input_size']
        self.input_size = params['input_size']
        self.output_size = params['output_size']
        self.center_size = params['num_modules'] * params['center_size_per_module']
        self.center_size_per_module = params['center_size_per_module']
        self.num_modules = params['num_modules']
        self.output_center_size =  self.output_size + self.center_size_per_module

        # This is for the time plot
        self.cell_state_history = None

        super(THALNET, self).__init__()

        # Create the DWM components
        self.ThalnetCell = ThalNetCell(self.input_size, self.output_size, self.context_input_size,
                                       self.center_size_per_module, self.num_modules)

    def forward(self, tuple):  # x : batch_size, seq_len, input_size
        """
        Runs the DWM cell and plots if necessary

        :param x: input sequence  [BATCH_SIZE x seq_len x input_size ]
        :param state: Input hidden state  [BATCH_SIZE x state_size]
        :return: Tuple [output, hidden_state]
        """
        (inputs, targets) = tuple

        output = None
        batch_size = inputs.size(0)
        seq_length = inputs.size(2)

        # init state
        cell_state = self.init_state(batch_size)
        for j in range(seq_length):
            print(inputs[..., j, :].size())
            output_cell, cell_state = self.ThalnetCell(inputs[..., j, :], cell_state)

            if output_cell is None:
                continue

            output_cell = output_cell[..., None, :]
            if output is None:
                output = output_cell

            # concatenate output
            else:
                output = torch.cat([output, output_cell], dim=-2)

        return output

    def init_state(self, batch_size):

        # center state initialisation
        center_state_per_module = [torch.randn((batch_size, self.center_size_per_module))
                                        for _ in range(self.num_modules)]

        # module state initialisation
        module_states = [torch.randn((batch_size, self.center_size_per_module if i != self.num_modules - 1 else self.output_center_size))
                         for i in range(self.num_modules)]

        states = center_state_per_module + module_states

        return states


if __name__ == "__main__":
    input_size = 28
    params = {'context_input_size': 32, 'input_size' : input_size, 'output_size': 10,
              'center_size': 1, 'center_size_per_module':32 , 'num_modules':4}

    model = THALNET(params)

    seq_length = 10
    batch_size = 2

    # Check for different seq_lengts and batch_sizes.
    for i in range(1):
        # Create random Tensors to hold inputs and outputs
        x = torch.randn(batch_size, 1, seq_length, input_size)

        # Test forward pass.
        y_pred = model(x)

        print(y_pred.size())

        # Change batch size and seq_length.
        seq_length = seq_length + 1
        batch_size = batch_size + 1



