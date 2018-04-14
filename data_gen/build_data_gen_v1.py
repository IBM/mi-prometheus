import numpy as np
import torch
from torch.autograd import Variable

CUDA = False
dtype = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


def my_print(name, var):
    print('=================')
    print(name)
    print(var)
    print('================')


def init_state(batch_size, tm_output_units, tm_state_units, n_heads, N, M):
    tm_output = Variable(torch.ones((batch_size, tm_output_units)).type(dtype))
    tm_state = Variable(torch.ones((batch_size, tm_state_units)).type(dtype))
    wt = Variable(torch.zeros((batch_size, n_heads, N)).type(dtype))
    wt[:, 0, 0] = 1.0

    mem_t = Variable((torch.ones((batch_size, M, N)) * 0.01).type(dtype))

    states = [tm_state, wt, tm_state,  mem_t]
    return tm_output, states


############################################################
#####                    Data generation            ########
############################################################


# now creating channel markers
pos = [0,0,0,0]
ctrl_data = [0,0,0,0]
ctrl_dummy = [1,0,0,0]
ctrl_inter = [0,0,0,1]


# add control channels to a sequence
def add_ctrl(seq, ctrl): return np.insert(seq, pos, ctrl, axis=-1)


# create augmented sequence as well as end marker and a dummy sequence
def augment(seq, ctrl_end):
    w = add_ctrl(seq, ctrl_data)
    end = add_ctrl(np.zeros((seq.shape[0], 1, seq.shape[2])), ctrl_end)
    dummy = add_ctrl(np.zeros_like(seq), ctrl_dummy)
    return [w, end, dummy]


def build_data_distraction(min_len, max_len, batch_size, bias, element_size, nb_markers_max):

    # Create a generator
    while True:
        # number of sub_sequences
        nb_sub_seq_a = np.random.randint(1, nb_markers_max)
        nb_sub_seq_b = nb_sub_seq_a              # might be different in future implementation

        # set the sequence length of each marker
        seq_lengths_a = np.random.randint(low=min_len, high=max_len + 1, size=nb_sub_seq_a)
        seq_lengths_b = np.random.randint(low=min_len, high=max_len + 1, size=nb_sub_seq_b)

        #  generate subsequences for x and y
        x = [np.random.binomial(1, bias, (batch_size, n, element_size)) for n in seq_lengths_a]
        y = [np.random.binomial(1, bias, (batch_size, n, element_size)) for n in seq_lengths_b]

        # create the target
        target = np.concatenate(y + x, axis=1)

        xx = [augment(seq, ctrl_end=[0,1,0,0]) for seq in x]
        yy = [augment(seq, ctrl_end=[0,0,1,0]) for seq in y]

        inter_seq = add_ctrl(np.zeros((batch_size, 1, element_size)), ctrl_inter)
        data_1 = [arr for a, b in zip(xx, yy) for arr in a[:-1] + b + [inter_seq]]
        data_2 = [a[-1] for a in xx]
        inputs = np.concatenate(data_1 + data_2, axis=1)

        inputs = Variable(torch.from_numpy(inputs).type(dtype))
        target = Variable(torch.from_numpy(target).type(dtype))
        mask = (inputs[0, :, 0] == 1)

        yield inputs, target, nb_sub_seq_a, mask

#########################################################################

a = build_data_distraction(3, 6, 1, 0.5, 8, 5)

for inputs, target, nb_marker, mask in a:
    print(mask)
    my_print('inputs', inputs)
    my_print('target', target)
    break







