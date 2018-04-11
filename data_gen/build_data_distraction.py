import torch
import pdb
from torch.autograd import Variable
import numpy as np
import sys

CUDA = False
dtype = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


def init_state(batch_size, tm_output_units, tm_state_units, n_heads, N, M):
    tm_output = Variable(torch.ones((batch_size, tm_output_units)).type(dtype))
    tm_state = Variable(torch.ones((batch_size, tm_state_units)).type(dtype))
    wt = Variable(torch.zeros((batch_size, n_heads, N)).type(dtype))
    wt[:, 0, 0] = 1.0

    mem_t = Variable((torch.ones((batch_size, M, N)) * 0.01).type(dtype))

    states = [tm_state, wt, mem_t]
    return tm_output, states


def build_data_gen(min_len, max_len, batch_size, bias, element_size, nb_markers_max):
    channel_length = 3
    dummy_size = element_size + channel_length

    while True:
        # number of sub_sequences
        nb_sub_seq_a = np.random.randint(1, nb_markers_max)
        nb_sub_seq_b = nb_sub_seq_a

        # set the sequence length of each marker
        seq_lengths_a = np.random.randint(low=min_len, high=max_len + 1, size= nb_sub_seq_a)
        seq_lengths_b = np.random.randint(low=min_len, high=max_len + 1, size= nb_sub_seq_b)

        print("x", seq_lengths_a)
        print("y", seq_lengths_b)

        # set the position of markers
        shift = np.arange(nb_sub_seq_a)
        position_markers_a = np.cumsum(seq_lengths_a[:-1] + seq_lengths_b[:-1])
        position_markers_a = np.append(position_markers_a, sum(seq_lengths_a)+sum(seq_lengths_b))

        position_markers_b = np.cumsum(seq_lengths_a[1:] + seq_lengths_b[:-1])
        position_markers_b = np.append(seq_lengths_a[0], position_markers_b+seq_lengths_a[0])

        # set values of marker
        marker_a = np.zeros((1, 1, element_size + channel_length))
        marker_b = np.zeros((1, 1, element_size + channel_length))

        marker_a[:, :, 0] = 1
        marker_b[:, :, 1] = 1

        # Create the sequence
        seq = np.random.binomial(1, bias, (batch_size, sum(seq_lengths_b)+sum(seq_lengths_a), element_size))

        # Add two channels
        inputs = np.insert(seq, (0, 0, 0), 0, axis=2)

        # Add markers
        if nb_sub_seq_a != 0:
            inputs = np.insert(inputs, tuple(position_markers_a), marker_a, axis=1)
            inputs = np.insert(inputs, tuple(position_markers_b)+shift, marker_b, axis=1)

        # insert_dummies for ys reading
        temp = 2
        for i in range(nb_sub_seq_a):
            shift_dummy = shift + temp
            dummy_y = np.zeros((batch_size, seq_lengths_b[i], dummy_size))
            dummy_y[:,:,2] = 1
            inputs = np.insert(inputs, [position_markers_a[i] + shift_dummy[i]], dummy_y, axis=1)
            j = i + 1
            temp = seq_lengths_b[i] + j + 2

        target = seq[:, position_markers_b[0]:position_markers_a[0], :]
        for i in range(1, nb_sub_seq_b):
            target = np.concatenate((target, seq[:, position_markers_b[i]:position_markers_a[i], :]),
                                    axis=1)

        target = np.concatenate((target, seq[:, :position_markers_b[0]:, :]), axis=1)
        for i in range(nb_sub_seq_a-1):
            target = np.concatenate((target, seq[:, position_markers_a[i]:position_markers_b[i+1], :]),
                                    axis=1)

        dummy_input = np.zeros((batch_size, sum(seq_lengths_a)+seq_lengths_b[-1], dummy_size))
        dummy_input[:, :, 2] = 1
        inputs = np.concatenate((inputs, dummy_input), axis=1)

        inputs = Variable(torch.from_numpy(inputs).type(dtype))
        target = Variable(torch.from_numpy(target).type(dtype))

        yield inputs, target, sum(seq_lengths_a)+seq_lengths_b[-1]

a = build_data_gen(3, 6, 1, 0.5, 8, 5)

for inputs, target, seq_length in a:
    print("inputs", inputs)
    print("target", target)
    break
