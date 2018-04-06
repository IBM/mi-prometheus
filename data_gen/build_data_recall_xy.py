import torch
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


def build_data_gen(min_len, max_len, batch_size, bias, element_size, nb_max):
    while True:
        seq_length = np.random.randint(low=min_len, high=max_len + 1)
        seq = np.random.binomial(1, bias, (batch_size, seq_length, element_size))
        encoder_input = np.insert(seq, 0, 0, axis=2)

        target = np.concatenate((seq[:, 0::2, :], seq[:, 1::2, :]), axis=1)
        dummy_size = element_size + 1
        dummy_input = np.zeros((batch_size, seq_length, dummy_size))
        dummy_input[:, :, 0] = 1
        inputs = np.concatenate((encoder_input, dummy_input), axis=1)

        inputs = torch.from_numpy(inputs).float()
        target = torch.from_numpy(target).float()

        yield inputs, target, seq_length

a = build_data_gen(3, 6, 1, 0.5, 8, 4)

for inputs, target, seq_length in a:
    print("seq_length", seq_length)
    print("inputs", inputs)
    print("target", target)
    break

