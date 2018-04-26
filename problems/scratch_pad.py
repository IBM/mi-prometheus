import torch
from torch.autograd import Variable
import numpy as np
from utils import augment
from algorithmic_sequential_problem import AlgorithmicSequentialProblem


@AlgorithmicSequentialProblem.register
class GeneratorScratchPad(AlgorithmicSequentialProblem):
    def __init__(self, params):
        # Retrieve parameters from the dictionary.
        self.batch_size = params['batch_size']
        # Number of bits in one element.
        self.control_bits = params['control_bits']
        self.data_bits = params['data_bits']
        assert self.control_bits >=2, "Problem requires at least 2 control bits (currently %r)" % self.control_bits
        assert self.data_bits >=1, "Problem requires at least 1 data bit (currently %r)" % self.data_bits
        # Min and max lengts (number of elements).
        self.min_sequence_length = params['min_sequence_length']
        self.max_sequence_length = params['max_sequence_length']
        # Number of subsequences.
        self.num_subseq_min = params["num_subseq_min"]
        self.num_subseq_max = params["num_subseq_max"]
        # Parameter  denoting 0-1 distribution (0.5 is equal).
        self.bias = params['bias']
        self.dtype = torch.FloatTensor

    def generate_batch(self,  seq_length):
        """Generates a batch  of size [BATCH_SIZE, ?, CONTROL_BITS+DATA_BITS].
       
        :param seq_length: the length of the copy sequence. (NOT USED NOW!)
        :returns: Tuple consisting of: input, output and mask

        TODO: deal with batch_size > 1
        """
        pos = [0, 0]
        ctrl_data = [0, 0]
        ctrl_dummy = [0, 1]

        markers = ctrl_data, ctrl_dummy, pos
        # Create a generator
        while True:
            # number sub sequences
            num_sub_seq = np.random.randint(self.num_subseq_min, self.num_subseq_max)

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

            inputs = Variable(torch.from_numpy(inputs).type(self.dtype))
            target = Variable(torch.from_numpy(target).type(self.dtype))
            mask = inputs[:, :, 1] == 1

            return inputs, target, mask


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    params = {'name': 'serial_recall_original', 'control_bits': 2, 'data_bits': 8, 'batch_size': 1,
              'min_sequence_length': 1, 'max_sequence_length': 10, 'bias': 0.5, 'num_subseq_min':1 ,'num_subseq_max': 4}
    # Create problem object.
    problem = GeneratorScratchPad(params)
    # Get generator
    generator = problem.return_generator_random_length()
    # Get batch.
    (x, y, mask) = next(generator)
    # Display single sample (0) from batch.
    problem.show_sample(x, y, mask)
