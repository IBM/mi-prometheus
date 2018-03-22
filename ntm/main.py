import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

from ntm.ntm_layer import NTM

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


def init_state(batch_size):
    tm_state = Variable(torch.ones((batch_size, tm_state_units))) * 0.01

    wt = Variable(torch.zeros((batch_size, n_heads, N)))
    wt[:, :, 0] = 1.0

    mem_t = Variable(torch.ones((batch_size, M, N))) * 0.01

    return tm_state, wt, mem_t


states = init_state(batch_size)
x = Variable(torch.ones(batch_size, seq, tm_in_dim))

ntm = NTM(tm_in_dim, tm_output_units, tm_state_units, n_heads, is_cam, num_shift, M)

output, states = ntm(x, states)
print("output", output, "memory", states[2])