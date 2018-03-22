import torch
from torch.autograd import Variable

import numpy as np


def init_state(batch_size, tm_output_units, tm_state_units, n_heads, N, M):
    tm_output = Variable(torch.ones((batch_size, tm_output_units)))
    tm_state = Variable(torch.ones((batch_size, tm_state_units)), requires_grad=True) * 0.01
    wt = torch.zeros((batch_size, n_heads, N))
    wt[:, :, 0] = 1.0

    wt = Variable(wt, requires_grad=True)
    mem_t = Variable(torch.ones((batch_size, M, N)), requires_grad=True) * 0.01

    states = [tm_state, wt, mem_t]
    return tm_output, states


def build_data_gen(min_len, max_len, batch_size, bias, element_size):
    seq_length = np.random.randint(low=min_len, high=max_len+1)
    inputs_encode = np.random.binomial(1, bias, (batch_size, seq_length, element_size))

    inputs = Variable(torch.from_numpy(inputs_encode).float(), requires_grad=True)
    target = Variable(torch.from_numpy(inputs_encode).float())

    return inputs, target, seq_length


