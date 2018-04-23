import torch
from torch.autograd import Variable
import numpy as np
from data_gen.utils import augment

CUDA = False
dtype = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


def data_generator(min_len, max_len, batch_size, bias, element_size, nb_seq_min, nb_seq_max):
    pos = [0, 0]
    ctrl_data = [0, 0]
    ctrl_dummy = [0, 1]

    markers = ctrl_data, ctrl_dummy, pos
    # Create a generator
    while True:
        # number sub sequences
        nb_sub_seq = np.random.randint(nb_seq_min, nb_seq_max)

        # set the sequence length of each marker
        seq_length = np.random.randint(low=min_len, high=max_len + 1, size=nb_sub_seq)

        #  generate subsequences for x and y
        x = [np.random.binomial(1, bias, (batch_size, n, element_size)) for n in seq_length]

        # create the target
        target = x[-1]

        xx = [augment(seq, markers, ctrl_end=[1,0], add_marker=True) for seq in x]

        data_1 = [arr for a in xx for arr in a[:-1]]
        data_2 = [xx[-1][-1]]

        inputs = np.concatenate(data_1+data_2, axis=1)

        inputs = Variable(torch.from_numpy(inputs).type(dtype))
        target = Variable(torch.from_numpy(target).type(dtype))
        mask = inputs[0, :, 1] == 1

        yield inputs, target, nb_sub_seq, mask

a = data_generator(3, 6, 1, 0.5, 8, 3, 4)

for inputs, target, _, mask in a:
    print(mask)
    print('inputs', inputs)
    print('target', target)
    break