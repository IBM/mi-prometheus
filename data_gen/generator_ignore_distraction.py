import numpy as np
import torch
from torch.autograd import Variable
from data_gen.utils import augment, add_ctrl

CUDA = False
dtype = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


def data_generator(min_len, max_len, batch_size, bias, element_size, nb_seq_min, nb_seq_max):
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
        target = np.concatenate([y[-1]] + x, axis=1)

        xx = [augment(seq, markers, ctrl_end=[1,0,0], add_marker=True) for seq in x]
        yy = [augment(seq, markers, ctrl_end=[0,1,0], add_marker=True) for seq in y]

        inter_seq = add_ctrl(np.zeros((batch_size, 1, element_size)), ctrl_inter, pos)
        data_1 = [arr for a, b in zip(xx, yy) for arr in a[:-1] + b[:-1]]


        # dummies of y and xs
        data_2 = [yy[-1][-1]] + [inter_seq] + [a[-1] for a in xx]
        inputs = np.concatenate(data_1 + data_2, axis=1)

        inputs = Variable(torch.from_numpy(inputs).type(dtype))
        target = Variable(torch.from_numpy(target).type(dtype))
        mask = inputs[0, :, 2] == 1

        yield inputs, target, nb_sub_seq_a, mask


a = data_generator(3, 6, 1, 0.5, 8, 3, 4)

for inputs, target, _, mask in a:
    print(mask)
    print('inputs', inputs)
    print('target', target)
    break







