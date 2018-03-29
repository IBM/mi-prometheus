import torch
from torch import nn
from ntm.build_data_copy import init_state, build_data_gen
import pdb
from ntm.ntm_layer import NTM
import numpy as np


# data generator x,y
batch_size = 1
min_len = 10
max_len = 20
bias = 0.5
element_size = 8
N = 30

# read training arguments
path = "/Users/younesbouhadajr/Documents/Neural_Network/working_memory/Models/"
read_arguments = np.load(path+"ntm_arguments.npy").item()

# init state, memory, attention
tm_in_dim = read_arguments['tm_in_dim']
tm_output_units = read_arguments['tm_output_units']
tm_state_units = read_arguments['tm_state_units']
n_heads = read_arguments['n_heads']
M = read_arguments['M']
is_cam = read_arguments['is_cam']
num_shift = read_arguments['num_shift']

# Test
print("Testing")

# New sequence
seed = 1
data_gen = build_data_gen(min_len, max_len, batch_size, bias, element_size)

# Instantiate
ntm = NTM(tm_in_dim, tm_output_units,tm_state_units, n_heads, is_cam, num_shift, M)

# Init state, memory, attention
_, states = init_state(batch_size, tm_output_units, tm_state_units, n_heads, N, M)

ntm.load_state_dict(torch.load(path+"model_parameters"))

for inputs, targets, seq_length in data_gen:
    output, states = ntm(inputs, states)
    print("output", output[:, -seq_length:, :])
    print("targets", targets)
    print("memory", states[2])

    # test accuracy
    output = torch.round(output[:, -seq_length:, :])
    acc = 1 - torch.abs(output-targets)
    accuracy = acc.mean()
    print("Accuracy: %.6f" % (accuracy * 100) + "%")
    break   # one test sample




