import torch
from torch.autograd import Variable
import numpy as np
from utils import augment, add_ctrl
from algorithmic_sequential_problem import AlgorithmicSequentialProblem


@AlgorithmicSequentialProblem.register
class GeneratorScratchPad(AlgorithmicSequentialProblem):
    """
    Class generating sequences of random bit-patterns and targets forcing the system to learn scratch pad problem (overwrite the memory).
    """

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

    def generate_batch(self):
        """Generates a batch  of size [BATCH_SIZE, ?, CONTROL_BITS+DATA_BITS].
       
        :returns: Tuple consisting of: input, output and mask
                  pattern of inputs: x1, x2, ...xn d
                  pattern of output: d, d,   ...d xn

        TODO: deal with batch_size > 1
        """
        pos = [0, 0]
        ctrl_data = [0, 0]
        ctrl_dummy = [0, 1]
        ctrl_inter = [0, 1]

        markers = ctrl_data, ctrl_dummy, pos

        # number sub sequences
        num_sub_seq = np.random.randint(self.num_subseq_min, self.num_subseq_max)

        # set the sequence length of each marker
        seq_length = np.random.randint(low=self.min_sequence_length, high=self.max_sequence_length + 1, size=num_sub_seq)

        #  generate subsequences for x and y
        x = [np.random.binomial(1, 0.5, (self.batch_size, n, self.data_bits)) for n in seq_length]

        # create the target
        seq_length_tdummies = sum(seq_length) + seq_length.shape[0] + 1
        dummies_target = np.zeros([self.batch_size, seq_length_tdummies, self.data_bits], dtype=np.float32)

        target = np.concatenate((dummies_target, x[-1]), axis=1)

        xx = [augment(seq, markers, ctrl_start=[1,0], add_marker_data=True, add_marker_dummy = False) for seq in x]

        data_1 = [arr for a in xx for arr in a[:-1]]
        inter_seq = add_ctrl(np.zeros((self.batch_size, 1, self.data_bits)), ctrl_inter, pos)
        data_2 = [xx[-1][-1]]

        inputs = np.concatenate(data_1 + [inter_seq] + data_2, axis=1)

        inputs = Variable(torch.from_numpy(inputs).type(self.dtype))
        target = Variable(torch.from_numpy(target).type(self.dtype))
        # TODO: batch might have different sequence lengths
        mask_all = inputs[..., 0:self.control_bits] == 1
        mask = mask_all[..., 0]
        for i in range(self.control_bits):
            mask = mask_all[..., i] * mask

        # TODO: fix the batch indexing
        # rest channel values of data dummies
        inputs[:, mask[0], 0:self.control_bits] = 0

        return inputs, target, mask


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    params = {'control_bits': 2, 'data_bits': 8, 'batch_size': 1,
              'min_sequence_length': 1, 'max_sequence_length': 10, 
              'bias': 0.5, 'num_subseq_min':1 ,'num_subseq_max': 4}
    # Create problem object.
    problem = GeneratorScratchPad(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    (x, y, mask) = next(generator)
    # Display single sample (0) from batch.
    problem.show_sample(x, y, mask)
