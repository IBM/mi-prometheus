import torch
from data_gen.generate_copy import data_generator
from data_gen.init_state import init_state
from ntm.ntm_layer import NTM
import numpy as np
import os

np.random.seed(999999999)

# read training arguments
path = "./Models/"
read_arguments = np.load(path+"ntm_arguments.npy").item()

# data_gen generator x,y
batch_size = 1
min_len = 10
max_len = 10
bias = 0.5
nb_markers_max = 5
nb_makers_min = 4
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
data_gen = data_generator(min_len, max_len, batch_size, bias, element_size, nb_makers_min, nb_markers_max)

# Instantiate
ntm = NTM(tm_in_dim, tm_output_units,tm_state_units, n_heads, is_cam, num_shift, M)

ntm.load_state_dict(torch.load(path+"model_parameters"))


for inputs, targets, nb_markers, mask in data_gen:

    # Init state, memory, attention
    N = 50 #max(seq_length)
    _, states = init_state(batch_size, tm_output_units, tm_state_units, n_heads, N, M)
    print('nb_markers', nb_markers)

    output, states = ntm(inputs, states)

    # test accuracy
    output = torch.round(output[:, mask, :])
    acc = 1 - torch.abs(output-targets)
    accuracy = acc.mean()
    print("Accuracy: %.6f" % (accuracy * 100) + "%")

    break   # one test sample




