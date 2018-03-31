import torch
from torch import nn
from ntm.build_data_scratch_pad import init_state, build_data_gen
import pdb
from ntm.ntm_layer import NTM
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(2)
np.random.seed(207)

# read training arguments
path = "/Users/younesbouhadajr/Documents/Neural_Network/working_memory/Models/"
read_arguments = np.load(path+"ntm_arguments.npy").item()

# data generator x,y
batch_size = 1
min_len = 50
max_len = 80
bias = 0.5
nb_markers_max = 5
element_size = read_arguments['element_size']

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
data_gen = build_data_gen(min_len, max_len, batch_size, bias, element_size, nb_markers_max)

# Instantiate
ntm = NTM(tm_in_dim, tm_output_units,tm_state_units, n_heads, is_cam, num_shift, M)

ntm.load_state_dict(torch.load(path+"model_parameters"))

for inputs, targets, seq_length in data_gen:

    # Init state, memory, attention
    N = max(seq_length)
    _, states = init_state(batch_size, tm_output_units, tm_state_units, n_heads, N, M)

    output, states = ntm(inputs, states)

    # test accuracy
    output = torch.round(output[:, -seq_length[-1]:, :])
    acc = 1 - torch.abs(output-targets)
    accuracy = acc.mean()
    print("Accuracy: %.6f" % (accuracy * 100) + "%")

    plt.ioff()
    plt.show()
    break   # one test sample




