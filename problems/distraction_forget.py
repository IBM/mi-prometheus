import numpy as np
import torch
from torch.autograd import Variable
from utils import augment, add_ctrl
from algorithmic_sequential_problem import AlgorithmicSequentialProblem


@AlgorithmicSequentialProblem.register
class DistractionForget(AlgorithmicSequentialProblem):
    """
    TODO: @Byounes: add task description.
    """
    def __init__(self, params):
        # Retrieve parameters from the dictionary.
        self.batch_size = params['batch_size']
        # Number of bits in one element.
        self.control_bits = params['control_bits']
        self.data_bits = params['data_bits']
        assert self.control_bits >=3, "Problem requires at least 3 control bits (currently %r)" % self.control_bits
        assert self.data_bits >=1, "Problem requires at least 1 data bit (currently %r)" % self.data_bits
        # Min and max lengts of a single subsequence (number of elements).
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

        TODO: deal with batch_size > 1
        """
        pos = [0, 0, 0]
        ctrl_data = [0, 0, 0]
        ctrl_dummy = [0, 0, 1]
        ctrl_inter = [1, 1, 0]

        markers = ctrl_data, ctrl_dummy, pos

        # number of sub_sequences
        nb_sub_seq_a = np.random.randint(self.num_subseq_min, self.num_subseq_max)
        nb_sub_seq_b = nb_sub_seq_a              # might be different in future implementation

        # set the sequence length of each marker
        seq_lengths_a = np.random.randint(low=self.min_sequence_length, high=self.max_sequence_length + 1, size=nb_sub_seq_a)
        seq_lengths_b = np.random.randint(low=self.min_sequence_length, high=self.max_sequence_length + 1, size=nb_sub_seq_b)

        #  generate subsequences for x and y
        x = [np.random.binomial(self.batch_size, self.bias, (self.batch_size, n, self.data_bits)) for n in seq_lengths_a]
        y = [np.random.binomial(self.batch_size, self.bias, (self.batch_size, n, self.data_bits)) for n in seq_lengths_b]

        # create the target
        target = np.concatenate(y + x, axis=1)
        
        xx = [augment(seq, markers, ctrl_start=[1,0,0], add_marker_data=True) for seq in x]
        yy = [augment(seq, markers, ctrl_start=[0,1,0], add_marker_data=True) for seq in y]

        inter_seq = add_ctrl(np.zeros((self.batch_size, 1, self.data_bits)), ctrl_inter, pos)
        data_1 = [arr for a, b in zip(xx, yy) for arr in a[:-1] + b]

        data_2 = [a[-1][:, 1:, :] for a in xx]
        inputs = np.concatenate(data_1 + [inter_seq] + data_2, axis=1)

        inputs = Variable(torch.from_numpy(inputs).type(self.dtype))
        target = Variable(torch.from_numpy(target).type(self.dtype))

        # create the mask
        mask_all = inputs[:, :, 0:self.control_bits] == 1
        mask = mask_all[..., 0]
        for i in range(self.control_bits):
            mask = mask_all[..., i] * mask

        # rest ctrl channel of dummies
        inputs[:, mask[0], 0:self.control_bits] = 0

        # Create the target with the dummies
        target_with_dummies = torch.zeros_like(inputs[:, :, self.control_bits:])
        target_with_dummies[:, mask[0], :] = target

        return inputs, target_with_dummies, mask


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    params = {'control_bits': 3, 'data_bits': 8, 'batch_size': 1,
              'min_sequence_length': 1, 'max_sequence_length': 10, 
              'bias': 0.5, 'num_subseq_min':1 ,'num_subseq_max': 4}
    # Create problem object.
    problem = DistractionForget(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    (x, y, mask) = next(generator)
    # Display single sample (0) from batch.
    problem.show_sample(x, y, mask)







