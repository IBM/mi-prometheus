import numpy as np
import torch
from torch.autograd import Variable
from problems.utils import augment, add_ctrl

CUDA = False
dtype = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


class GenerateForgetDistraction(object):
    def __init__(self, params):
        self.min_sequence_length = params["min_sequence_length"]
        self.max_sequence_length = params["max_sequence_length"]
        self.num_seq_min = params["num_seq_min"]
        self.num_seq_max = params["num_seq_max"]
        self.batch_size = params["batch_size"]
        self.data_bits = params["data_bits"]

    def data_generator(self):
        pos = [0, 0, 0]
        ctrl_data = [0, 0, 0]
        ctrl_dummy = [0, 0, 1]
        ctrl_inter = [1, 1, 0]

        markers = ctrl_data, ctrl_dummy, pos

        # Create a generator
        while True:
            # number of sub_sequences
            nb_sub_seq_a = np.random.randint(self.num_seq_min, self.num_seq_max)
            nb_sub_seq_b = nb_sub_seq_a              # might be different in future implementation

            # set the sequence length of each marker
            seq_lengths_a = np.random.randint(low=self.min_sequence_length, high=self.max_sequence_length + 1, size=nb_sub_seq_a)
            seq_lengths_b = np.random.randint(low=self.min_sequence_length, high=self.max_sequence_length + 1, size=nb_sub_seq_b)

            #  generate subsequences for x and y
            x = [np.random.binomial(1, 0.5, (self.batch_size, n, self.data_bits)) for n in seq_lengths_a]
            y = [np.random.binomial(1, 0.5, (self.batch_size, n, self.data_bits)) for n in seq_lengths_b]

            # create the target
            target = np.concatenate(y + x, axis=1)

            xx = [augment(seq, markers, ctrl_end=[1,0,0], add_marker=True) for seq in x]
            yy = [augment(seq, markers, ctrl_end=[0,1,0], add_marker=True) for seq in y]

            inter_seq = add_ctrl(np.zeros((self.batch_size, 1, self.data_bits)), ctrl_inter, pos)
            data_1 = [arr for a, b in zip(xx, yy) for arr in a[:-1] + b + [inter_seq]]

            data_2 = [a[-1] for a in xx]
            inputs = np.concatenate(data_1 + data_2, axis=1)

            inputs = Variable(torch.from_numpy(inputs).type(dtype))
            target = Variable(torch.from_numpy(target).type(dtype))
            mask = inputs[0, :, 2] == 1

            yield inputs, target, mask


params = {"min_sequence_length":1,
         "max_sequence_length":4,
         "batch_size":1,
         "data_bits": 8,
         "num_seq_min":1,
         "num_seq_max": 4}

data = GenerateForgetDistraction(params)

for inputs, target, mask in data.data_generator():
    print(mask)
    print('inputs', inputs)
    print('target', target)
    break







