import torch
from torch.autograd import Variable
import pdb

import numpy as np


def init_state(batch_size, tm_output_units, tm_state_units, n_heads, N, M):
    tm_output = torch.ones((batch_size, tm_output_units))
    tm_state = torch.ones((batch_size, tm_state_units))
    wt = torch.zeros((batch_size, n_heads, N))
    wt[:, :, 0] = 1.0

    mem_t = torch.ones((batch_size, M, N)) * 0.01

    states = [tm_state, wt, mem_t]
    return tm_output, states


def data_gen_type1(min_len, max_len, batch_size, bias, element_size):
    dummy_size = element_size + 2

    seq_length = np.random.randint(low=min_len, high=max_len + 1)
    seq = np.random.binomial(1, bias, (batch_size, seq_length, element_size))
    inputs = np.insert(seq, (0,0), (1,1), axis=2)

    target = seq

    dummy_input = np.zeros((batch_size, seq_length, dummy_size))
    inputs = np.concatenate((inputs, dummy_input), axis=1)

    inputs = torch.from_numpy(inputs).float()
    target = torch.from_numpy(target).float()

    yield inputs, target, seq_length


def data_gen_type2(min_len, max_len, batch_size, bias, element_size):
    dummy_size = element_size + 2

    seq_length = np.random.randint(low=min_len, high=max_len + 1)
    seq = np.random.binomial(1, bias, (batch_size, 2*seq_length, element_size))
    input_marker = np.insert(seq, seq_length, 0, axis=1)
    inputs = np.insert(input_marker, (0,0), (1,1), axis=2)
    # put a separator
    inputs[:, seq_length, 1] = 0

    target = seq[:, -seq_length:, :]

    dummy_input = np.zeros((batch_size, seq_length, dummy_size))
    inputs = np.concatenate((inputs, dummy_input), axis=1)

    inputs = torch.from_numpy(inputs).float()
    target = torch.from_numpy(target).float()

    yield inputs, target, seq_length


def build_data_gen(min_len, max_len, batch_size, bias, element_size):
    selector = np.random.randint(0, 2)
    if selector:
        return data_gen_type1(min_len, max_len, batch_size, bias, element_size)
    else:
        return data_gen_type2(min_len, max_len, batch_size, bias, element_size)


