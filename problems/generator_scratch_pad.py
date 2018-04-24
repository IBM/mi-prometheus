import torch
from torch.autograd import Variable
import numpy as np
from problems.utils import augment

CUDA = False
dtype = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

class generator_scratch_pad(object):
    def __init__(self, params):
        self.min_sequence_length = params["min_sequence_length"]
        self.max_sequence_length = params["max_sequence_length"]
        self.num_seq_min = params["num_seq_min"]
        self.num_seq_max = params["num_seq_max"]
        self.batch_size = params["batch_size"]
        self.data_bits = params["data_bits"]

    def data_generator(self):
        pos = [0, 0]
        ctrl_data = [0, 0]
        ctrl_dummy = [0, 1]

        markers = ctrl_data, ctrl_dummy, pos
        # Create a generator
        while True:
            # number sub sequences
            num_sub_seq = np.random.randint(self.num_seq_min, self.num_seq_max)

            # set the sequence length of each marker
            seq_length = np.random.randint(low=self.min_sequence_length, high=self.max_sequence_length + 1, size=num_sub_seq)

            #  generate subsequences for x and y
            x = [np.random.binomial(1, 0.5, (self.batch_size, n, self.data_bits)) for n in seq_length]

            # create the target
            target = x[-1]

            xx = [augment(seq, markers, ctrl_end=[1,0], add_marker=True) for seq in x]

            data_1 = [arr for a in xx for arr in a[:-1]]
            data_2 = [xx[-1][-1]]

            inputs = np.concatenate(data_1+data_2, axis=1)

            inputs = Variable(torch.from_numpy(inputs).type(dtype))
            target = Variable(torch.from_numpy(target).type(dtype))
            mask = inputs[0, :, 1] == 1

            yield inputs, target, mask

params = {"min_sequence_length":1,
         "max_sequence_length":4,
         "batch_size":1,
         "data_bits": 8,
         "num_seq_min":1,
         "num_seq_max": 4}

data = generator_scratch_pad(params)

for inputs, target, mask in data.data_generator():
    print(mask)
    print('inputs', inputs)
    print('target', target)
    break