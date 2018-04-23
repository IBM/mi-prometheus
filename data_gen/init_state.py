import torch
from torch.autograd import Variable

CUDA = False
dtype = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


def init_state(batch_size, tm_output_units, tm_state_units, n_heads, N, M):
    tm_output = Variable(torch.ones((batch_size, tm_output_units)).type(dtype))
    tm_state = Variable(torch.ones((batch_size, tm_state_units)).type(dtype))

    # initial attention  vector
    wt = Variable(torch.zeros((batch_size, n_heads, N)).type(dtype))
    wt[:, 0:n_heads, 0] = 1.0

    # bookmark
    wt_dynamic = wt

    mem_t = Variable((torch.ones((batch_size, M, N)) * 0.01).type(dtype))

    states = [tm_state, wt, wt_dynamic, mem_t]
    return tm_output, states