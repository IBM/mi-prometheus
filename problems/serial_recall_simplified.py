import torch
from torch.autograd import Variable
import numpy as np
from problems.utils import augment
from problems.algorithmic_sequential_problem import AlgorithmicSequentialProblem


@AlgorithmicSequentialProblem.register
class SerialRecallSimplifiedProblem(AlgorithmicSequentialProblem):
    def __init__(self, params):
        self.min_sequence_length = params["min_sequence_length"]
        self.max_sequence_length = params["max_sequence_length"]
        self.batch_size = params["batch_size"]
        self.data_bits = params["data_bits"]
        self.dtype = torch.FloatTensor

    def generate_batch(self):
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

            # add dummies and markers to the sub sequence 
            xx = augment(x, markers)

            inputs = np.concatenate(xx, axis=1)

            inputs = Variable(torch.from_numpy(inputs).type(self.dtype))
            target = Variable(torch.from_numpy(target).type(self.dtype))

            # create a mask for the target
            mask = inputs[0, :, 1] == 1

            return inputs, target, mask


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    params = {'name': 'serial_recall_original', 'control_bits': 2, 'data_bits': 8, 'batch_size': 1,
              'min_sequence_length': 1, 'max_sequence_length': 10, 'bias': 0.5}
    # Create problem object.
    problem = SerialRecallSimplifiedProblem(params)
    # Get generator
    generator = problem.return_generator_random_length()
    # Get batch.
    (x, y, mask) = next(generator)
    # Display single sample (0) from batch.
    problem.show_sample(x, y, mask)