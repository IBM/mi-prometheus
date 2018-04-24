import torch
from torch.autograd import Variable
import numpy as np
from problems.utils import augment

CUDA = False
dtype = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


class generate_serial_recall(object):
    def __init__(self, params):
        self.min_sequence_length = params["min_sequence_length"]
        self.max_sequence_length = params["max_sequence_length"]
        self.batch_size = params["batch_size"]
        self.data_bits = params["data_bits"]

    def data_generator(self):
        pos = [0, 0]
        ctrl_data = [0, 0]
        ctrl_dummy = [0, 1]

        markers = ctrl_data, ctrl_dummy, pos
        # Create a generator
        while True:
            # set the sequence length of each marker
            seq_length = np.random.randint(low=self.min_sequence_length, high=self.max_sequence_length + 1)

            #  generate subsequences for x and y
            x = np.random.binomial(1, 0.5, (self.batch_size, seq_length, self.data_bits))

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

for inputs, target, mask in data.data_generator():
    print(mask)
    print('inputs', inputs)
    print('target', target)
    break