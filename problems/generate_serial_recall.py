import torch
from torch.autograd import Variable
import numpy as np
from problems.utils import augment

CUDA = False
dtype = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


def generate_serial_recall(params):
    pos = [0, 0]
    ctrl_data = [0, 0]
    ctrl_dummy = [0, 1]

    min_sequence_length = params["min_sequence_length"]
    max_sequence_length = params["max_sequence_length"]
    batch_size = params["batch_size"]
    data_bits = params["data_bits"]

    markers = ctrl_data, ctrl_dummy, pos
    # Create a generator
    while True:
        # set the sequence length of each marker
        seq_length = np.random.randint(low=min_sequence_length, high=max_sequence_length + 1)

        #  generate subsequences for x and y
        x = np.random.binomial(1, 0.5, (batch_size, seq_length, data_bits))

        # create the target
        target = x

        xx = augment(x, markers)

        inputs = np.concatenate(xx, axis=1)

        inputs = Variable(torch.from_numpy(inputs).type(dtype))
        target = Variable(torch.from_numpy(target).type(dtype))
        mask = inputs[0, :, 1] == 1
        nb_seq = 1          # I added this just so we have the same main
                            # for this task as well

        yield inputs, target, mask

params = {"min_sequence_length":1,
         "max_sequence_length":4,
         "batch_size":1,
         "data_bits": 8}

data = generate_serial_recall(params)

for inputs, target, mask in data:
    print(mask)
    print('inputs', inputs)
    print('target', target)
    break