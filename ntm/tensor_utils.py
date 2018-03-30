import torch
import torch.nn.functional as F
import numpy as np

# All input tensors are assumed to have hidden shapes (called "batch")
# as the first few dimensions


# Normalize last dimension with fuzzy factor
def normalize(x):
    return x / torch.max(torch.sum(x, dim=-1, keepdim=True), torch.Tensor([1e-12]))

# Batch cross-product similarity computed using matrix multiplication
# the hidden shapes must be broadcastable (numpy style)
def sim(query, data, l2_normalize=False, aligned=True):
    # data.shape = hidden_shape_1 x M x N
    # query.shape = hidden_shape_2 x h x p, where:
    #        p = N if aligned is True and p = M if aligned is False
    # out[...,i,j] = sum_k q[...,i,k] * data[...,j,k] for the default options

    if aligned:  # transpose last 2 dims to enable matrix multiplication
        data = torch.transpose(data, -1, -2)

    assert query.size()[-1] == data.size()[-2]

    if l2_normalize:
        query = F.normalize(query, dim=-1)
        data = F.normalize(data, dim=-2)

    return torch.matmul(query, data)


# Batch outer product of two vectors
# the hidden shapes must be broadcastable (numpy style)
def outer_prod(x, y):
    return x[..., :, None] * y[..., None, :]


# Batch 1D convolution with matching hidden shapes
def circular_conv(x, f):
    # computes y[...,i] = sum_{j=-ceil(s/2)+1}^{floor(s/2)} x[...,i-j] * f[...,j]

    f_last = f.size()[-1]
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
        y[ix] = F.conv1d(x[ix][None, None, :], f[ix][None, None, :])
    return y


def sharpen(ws, γ):
    #print("before", ws)
    #print(torch.sum(ws, -1).view(-1,1))
    w = ws ** γ
    w = torch.div(w, torch.sum(w, -1).view(-1,1) + 1e-16)
    #print("after", w)
    #input("pass")
    return w