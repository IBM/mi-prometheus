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
        # set the sequence length of each marker
        seq_length = np.random.randint(low=min_len, high=max_len + 1, )

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

a = data_generator(3, 6, 1, 0.5, 8, 3, 4)

for inputs, target, _, mask in a:
    print(mask)
    print('inputs', inputs)
    print('target', target)
    break