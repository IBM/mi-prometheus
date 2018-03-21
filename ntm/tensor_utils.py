import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


# Batch normalize last dimension with fuzzy factor
def normalize(x):
    return F.normalize(x, p=1, dim=-1)


# batch similarity
def sim(query, data, l2_normalize=False, transpose=False):
    # query.shape = hidden_shape_1 x h x M, if transpose is False
    # query.shape = hidden_shape_1 x h x N, if transpose is True
    # data.shape = hidden_shape_2 x M x N
    # the hidden shapes must be broadcastable (numpy style)
    # out[...,i,j] = sum_k q[...,i,k] * data[...,k,j] for the default options
    if transpose:
        data = torch.transpose(data, -1, -2)

    assert query.size()[-1] == data.size()[-2]

    if l2_normalize:
        query = F.normalize(query, dim=-1)
        data = F.normalize(data, dim=-2)

    return torch.matmul(query, data)


# Batch outer product of two vectors
def outer_prod(x, y):
    return x[..., :, None] * y[..., None, :]


# Batch 1D convolution of unequal length vectors
def circular_conv(x, f):
    # computes y[...,i] = sum_{j=-ceil(s/2)+1}^{floor(s/2)} x[...,i-j] * f[...,j]

    f_last = f.size()[-1]
    print("f_size:", f.size(), "x.size", x.size())
    assert (f_last >= 3) and (f_last <= x.size()[-1]), "filter size constraint violated"

    f_other = f.size()[:-1]
    assert f_other == x.size()[:-1], "hidden shapes should match"

    y = x.clone()
    ind_left = f_last // 2
    ind_right = f_last - ind_left - 1
    # padding to wrap x with itself
    x = torch.cat([x[..., -ind_left:], x, x[..., :ind_right]], dim=-1)

    # loop over indices in the hidden shape
    for ix in np.ndindex(f_other):
        print("ix",ix)
        y[ix] = F.conv1d(x[ix[0], ix[1], :][None, None, :], f[ix[0], ix[1], :][None, None, :])

    return y
