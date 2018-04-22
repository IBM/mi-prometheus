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
    # initial attention  vector
    wt = Variable(torch.zeros((batch_size, n_heads, N)).type(dtype))
    wt[:, 0:n_heads, 0] = 1.0
    # bookmark
    wt_dynamic = wt

    mem_t = Variable((torch.ones((batch_size, M, N)) * 0.01).type(dtype))

    states = [tm_state, wt, wt_dynamic, mem_t]
    return tm_output, states


############################################################
#####                    Data generation            ########
############################################################

# add control channels to a sequence
def add_ctrl(seq, ctrl, pos): return np.insert(seq, pos, ctrl, axis=-1)

# create augmented sequence as well as end marker and a dummy sequence
def augment(seq, markers, ctrl_end=None, add_marker=False):
    ctrl_data, ctrl_dummy, pos = markers

    w = add_ctrl(seq, ctrl_data, pos)
    end = add_ctrl(np.zeros((seq.shape[0], 1, seq.shape[2])), ctrl_end, pos)
    if add_marker:
        w = np.concatenate((w, end), axis=1)
    dummy = add_ctrl(np.zeros_like(seq), ctrl_dummy, pos)

    return [w, dummy]


def generate_forget_distraction(min_len, max_len, batch_size, bias, element_size, nb_seq_min, nb_seq_max):
    pos = [0, 0, 0]
    ctrl_data = [0, 0, 0]
    ctrl_dummy = [0, 0, 1]
    ctrl_inter = [1, 1, 0]

    markers = ctrl_data, ctrl_dummy, pos

    # Create a generator
    while True:
        # number of sub_sequences
        nb_sub_seq_a = np.random.randint(nb_seq_min, nb_seq_max)
        nb_sub_seq_b = nb_sub_seq_a              # might be different in future implementation

        # set the sequence length of each marker
        seq_lengths_a = np.random.randint(low=min_len, high=max_len + 1, size=nb_sub_seq_a)
        seq_lengths_b = np.random.randint(low=min_len, high=max_len + 1, size=nb_sub_seq_b)

        #  generate subsequences for x and y
        x = [np.random.binomial(1, bias, (batch_size, n, element_size)) for n in seq_lengths_a]
        y = [np.random.binomial(1, bias, (batch_size, n, element_size)) for n in seq_lengths_b]

        # create the target
        target = np.concatenate(y + x, axis=1)

        xx = [augment(seq, markers, ctrl_end=[1,0,0], add_marker=True) for seq in x]
        yy = [augment(seq, markers, ctrl_end=[0,1,0], add_marker=True) for seq in y]

        inter_seq = add_ctrl(np.zeros((batch_size, 1, element_size)), ctrl_inter, pos)
        data_1 = [arr for a, b in zip(xx, yy) for arr in a[:-1] + b + [inter_seq]]

        data_2 = [a[-1] for a in xx]
        inputs = np.concatenate(data_1 + data_2, axis=1)

        inputs = Variable(torch.from_numpy(inputs).type(dtype))
        target = Variable(torch.from_numpy(target).type(dtype))
        mask = inputs[0, :, 2] == 1

        yield inputs, target, nb_sub_seq_a, mask


def generate_copy(min_len, max_len, batch_size, bias, element_size, nb_seq_min, nb_seq_max):
    pos = [0, 0]
    ctrl_data = [0, 0]
    ctrl_dummy = [0, 1]

    markers = ctrl_data, ctrl_dummy, pos
    # Create a generator
    while True:
        # set the sequence length of each marker
        seq_length = np.random.randint(low=min_len, high=max_len + 1)

        #  generate subsequences for x and y
        x = np.random.binomial(1, bias, (batch_size, seq_length, element_size))

        # create the target
        target = x

        xx = augment(x, markers)

        inputs = np.concatenate(xx, axis=1)

        inputs = Variable(torch.from_numpy(inputs).type(dtype))
        target = Variable(torch.from_numpy(target).type(dtype))
        mask = inputs[0, :, 1] == 1
        nb_seq = 1          # I added this just so we have the same main
                            # for this task as well

        yield inputs, target, nb_seq, mask


a = generate_copy(3, 6, 1, 0.5, 8, 3, 4)

for inputs, target, _, mask in a:
    print(mask)
    my_print('inputs', inputs)
    my_print('target', target)
    break







