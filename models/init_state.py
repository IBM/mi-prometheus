import torch
from torch.autograd import Variable

CUDA = False
dtype = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


def init_state(params):

    batch_size = params["batch_size"]
    tm_output_units = params["data_bits"]
    tm_state_units = params["hidden_state_dim"]
    n_heads = params["num_heads"]
    num_shift = params["shift_size"]
    N = params["memory_addresses_size"]
    M = params["memory_content_size"]

    tm_output = Variable(torch.ones((batch_size, tm_output_units)).type(dtype))
    tm_state = Variable(torch.ones((batch_size, tm_state_units)).type(dtype))

    # initial attention  vector
    wt = Variable(torch.zeros((batch_size, n_heads, N)).type(dtype))
    wt[:, 0:n_heads, 0] = 1.0

    # bookmark
    wt_dynamic = wt

    mem_t = Variable((torch.ones((batch_size, M, N)) * 0.01).type(dtype))

    states = [tm_state, wt, wt_dynamic, mem_t]
    return states