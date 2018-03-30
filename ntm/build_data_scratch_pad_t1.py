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


def build_data_gen(min_len, max_len, batch_size, bias, element_size):
    dummy_size = element_size + 2
    while True:
        selector = np.random.randint(0, 2)
        if selector: # type 1
            seq_length = np.random.randint(low=min_len, high=max_len + 1)
            seq = np.random.binomial(1, bias, (batch_size, seq_length, element_size))
            inputs = np.insert(seq, (0, 0), (1, 1), axis=2)

            target = seq

            dummy_input = np.zeros((batch_size, seq_length, dummy_size))
            inputs = np.concatenate((inputs, dummy_input), axis=1)

            inputs = torch.from_numpy(inputs).float()
            target = torch.from_numpy(target).float()

            yield inputs, target, seq_length
        else: # type 2
            seq_len_1 = np.random.randint(low=min_len, high=max_len + 1)
            seq_len_2 = np.random.randint(low=min_len, high=max_len + 1)

            seq = np.random.binomial(1, bias, (batch_size, seq_len_1+seq_len_2, element_size))
            inputs_with_marker = np.insert(seq, seq_len_1, 0, axis=1)

            # add two channels
            inputs = np.insert(inputs_with_marker, (0, 0), (1, 1), axis=2)

            # set the channel values of separator
            inputs[:, seq_len_1, 1] = 0

            target = seq[:, -seq_len_2:, :]

            dummy_input = np.zeros((batch_size, seq_len_2, dummy_size))
            inputs = np.concatenate((inputs, dummy_input), axis=1)

            inputs = torch.from_numpy(inputs).float()
            target = torch.from_numpy(target).float()

            yield inputs, target, seq_len_2



#a = build_data_gen(3, 6, 2, 0.5, 5)


#for inputs, target, seq_length in a:
#    print(inputs)
#    print(target)
