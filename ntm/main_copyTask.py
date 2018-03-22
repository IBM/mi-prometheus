import sys
import torch
from torch import nn
from torch.autograd import Variable
from ntm.build_data import init_state, build_data_gen
import numpy as np

from ntm.ntm_layer import NTM

# data generator x,y
batch_size = 1
min_len = 3
max_len = 3
bias = 0.5
element_size = 5

# init state, memory, attention
tm_in_dim = element_size
tm_output_units = element_size
tm_state_units = 4
n_heads = 1
N = 6
M = 3
is_cam = False
num_shift = 3

# Instantiate
ntm = NTM(tm_in_dim, tm_output_units,tm_state_units, n_heads, is_cam, num_shift, M)

'''
tm_in_dim = 8
seq = 7
tm_output_units = 3
tm_state_units = 5
batch_size = 1
n_heads = 2
N = 10
M = 6
is_cam = False
num_shift = 3
'''

# Init state, memory, attention
tm_output, states = init_state(batch_size, tm_output_units, tm_state_units, n_heads, N, M)

# Data generator : input & target
inputs, targets, seq_length = build_data_gen(min_len, max_len, batch_size, bias, element_size)

# Set loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(ntm.parameters(), lr=0.01)
loss = 0

# Training
for epoch in range(4):
    optimizer.zero_grad()
    loss = 0

    sys.stdout.write("predicted string: ")
    output, states = ntm(inputs, states)
    #print("state",states[0], "wt",states[1], "memory", states[2])
    #print("output", output)
    #print("targets", targets)
    #input("pass")

    #print(output)
    #print(targets)
    #input("pass")
    loss = criterion(output, targets)

    print(", epoch: %d, loss: %1.3f" % (epoch + 1, loss.data[0]))

    loss.backward(retain_graph=True)

    optimizer.step()


print("Learning finished!")

# Test
print("Testing")
print("input")
