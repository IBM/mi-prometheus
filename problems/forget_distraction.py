import numpy as np
import torch
from torch.autograd import Variable
from problems.utils import augment, add_ctrl
from problems.algorithmic_sequential_problem import AlgorithmicSequentialProblem


@AlgorithmicSequentialProblem.register
class GenerateForgetDistraction(AlgorithmicSequentialProblem):
    def __init__(self, params):
        self.min_sequence_length = params["min_sequence_length"]
        self.max_sequence_length = params["max_sequence_length"]
        self.num_subseq_min = params["num_subseq_min"]
        self.num_subseq_max = params["num_subseq_max"]
        self.batch_size = params["batch_size"]
        self.data_bits = params["data_bits"]
        self.dtype = torch.FloatTensor


    def generate_batch(self):
        pos = [0, 0, 0]
        ctrl_data = [0, 0, 0]
        ctrl_dummy = [0, 0, 1]
        ctrl_inter = [1, 1, 0]

        markers = ctrl_data, ctrl_dummy, pos

        # Create a generator
        while True:
            # number of sub_sequences
            nb_sub_seq_a = np.random.randint(self.num_subseq_min, self.num_subseq_max)
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

            inputs = Variable(torch.from_numpy(inputs).type(self.dtype))
            target = Variable(torch.from_numpy(target).type(self.dtype))
            mask = inputs[0, :, 2] == 1

            return inputs, target, mask


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    params = {'name': 'serial_recall_original', 'control_bits': 2, 'data_bits': 8, 'batch_size': 1,
              'min_sequence_length': 1, 'max_sequence_length': 10, 'bias': 0.5, 'num_subseq_min':1 ,'num_subseq_max': 4}
    # Create problem object.
    problem = GenerateForgetDistraction(params)
    # Get generator
    generator = problem.return_generator_random_length()
    # Get batch.
    (x, y, mask) = next(generator)
    # Display single sample (0) from batch.
    problem.show_sample(x, y, mask)







