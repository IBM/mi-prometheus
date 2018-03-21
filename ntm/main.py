import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

from ntm.ntm_layer import NTM

tm_in_dim = 2
seq = 3
tm_output_units = 2
tm_state_units = 2
batch_size = 1
n_heads = 1
N = 3
M = 3
is_cam = 0
num_shift = 3

def init_state(batch_size):
    tm_output = Variable(torch.ones((batch_size, tm_output_units)))
    tm_state = Variable(torch.ones((batch_size, tm_state_units))) * 0.01

    wt = Variable(torch.zeros((batch_size, n_heads, N)))
    wt[:, :, 0] = 1.0

    mem_t = Variable(torch.ones((batch_size, N, M))) * 0.01

    return tm_output, tm_state, wt, mem_t

states = init_state(batch_size)[1:]
x = Variable(torch.ones(batch_size, seq, tm_in_dim))

ntm = NTM(tm_in_dim, tm_output_units,tm_state_units, n_heads, is_cam, num_shift, M)

output, states = ntm(x, states)
print("output", output, "memory", states[2])