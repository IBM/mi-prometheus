import torch
from torch import nn
from torch.autograd import Variable
from misc.app_state import AppState
from models.model_base import ModelBase


class LSTM(ModelBase, nn.Module):
    def __init__(self, params):
        self.tm_in_dim = params["control_bits"] + params["data_bits"]
        self.data_bits = params["data_bits"]
        self.hidden_state_dim = params["hidden_state_dim"]
        self.num_layers = params["num_layers"]
        assert self.num_layers > 0, "Number of LSTM layers should be > 0"
        self.app_state = AppState()

        super(LSTM, self).__init__()

        # Create the LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(nn.LSTMCell(self.tm_in_dim, self.hidden_state_dim))
        self.lstm_layers.extend([nn.LSTMCell(self.hidden_state_dim, self.hidden_state_dim)
                                 for _ in range(1, self.num_layers)])

        self.linear = nn.Linear(self.hidden_state_dim, self.data_bits)

    def forward(self, x):
        # Check if the class has been converted to cuda (through .cuda() method)
        dtype = torch.cuda.FloatTensor if next(self.linear.parameters()).is_cuda else torch.FloatTensor

        # Create the hidden state tensors
        h = [Variable(torch.zeros(x.size(0), self.hidden_state_dim).type(dtype), requires_grad=False)
                  for _ in range(self.num_layers)]

        # Create the internal state tensors
        c = [Variable(torch.zeros(x.size(0), self.hidden_state_dim).type(dtype), requires_grad=False)
                  for _ in range(self.num_layers)]

        outputs = []

        for x_t in x.chunk(x.size(1), dim=1):
            h[0], c[0] = self.lstm_layers[0](x_t.squeeze(1), (h[0], c[0]))
            for i in range(1, self.num_layers):
                h[i], c[i] = self.lstm_layers[i](h[i-1], (h[i], c[i]))

            out = self.linear(h[-1])
            outputs += [out]

        outputs = torch.stack(outputs, 1)
        return outputs

