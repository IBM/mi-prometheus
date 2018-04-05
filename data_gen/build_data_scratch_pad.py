import torch
import pdb

import numpy as np


def init_state(batch_size, tm_output_units, tm_state_units, n_heads, N, M):
    rN = 60
    tm_output = torch.ones((batch_size, tm_output_units))
    tm_state = torch.ones((batch_size, tm_state_units))
    wt = torch.zeros((batch_size, n_heads, N))
    wt[:, 0, 0] = 1.0
    wt[:, 1, rN] = 1.0
    mem_t = torch.ones((batch_size, M, N)) * 0.01

    states = [tm_state, wt, mem_t]
    return tm_output, states


def build_data_gen(min_len, max_len, batch_size, bias, element_size, nb_markers_max):
    channel_length = 3
    dummy_size = element_size + channel_length

    while True:
        # number of sub_sequences
        nb_markers = np.random.randint(0, nb_markers_max+1)

        # number of elements to be recalled
        nb_recall = nb_markers

        # set the sequence length of each marker
        seq_lengths = np.random.randint(low=min_len, high=max_len + 1, size=nb_markers+1)

        # set the position of markers
        shift = np.arange(nb_markers+1)
        position_recall = np.cumsum(seq_lengths)
        position_markers_b = position_recall + shift
        position_markers_a = position_recall + 2*shift + 2

        # set values of marker
        marker_a = np.zeros((1, 1, element_size + channel_length))
        marker_b = np.zeros((1, 1, element_size + channel_length))

        marker_a[:, :, 0] = 1
        marker_b[:, :, 1] = 1

        # Create the sequence
        seq = np.random.binomial(1, bias, (batch_size, sum(seq_lengths)+nb_recall, element_size))
        recall = seq[:, position_markers_b[:-1], :]

        # Add two channels
        inputs = np.insert(seq, (0, 0, 0), 0, axis=2)
        #inputs[:, position_markers_b[:-1], 0] = 1

        # Add markers
        if nb_markers != 0:
            position = tuple(position_markers_b[:-1]) + tuple(position_markers_a[:-1])
            inputs = np.insert(inputs, tuple(position_markers_b[:-1]), marker_a, axis=1)
            inputs = np.insert(inputs, tuple(position_markers_a[:-1]), marker_b, axis=1)

        target = np.concatenate((seq[:, -seq_lengths[-1]:, :], recall), axis=1)

        dummy_input = np.zeros((batch_size, seq_lengths[-1]+nb_recall, dummy_size))
        dummy_input[:, :, 2] = 1
        inputs = np.concatenate((inputs, dummy_input), axis=1)

        inputs = torch.from_numpy(inputs).float()
        target = torch.from_numpy(target).float()
        #print("seq_length:", seq_lengths)
        #print("nb_markers:", nb_markers)

        yield inputs, target, seq_lengths

a = build_data_gen(3, 6, 1, 0.5, 8, 5)

for inputs, target, seq_length in a:
    print("seq_length", seq_length)
    print("inputs", inputs)
    print("target", target)
    break
