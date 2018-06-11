import torch
import numpy as np
from utils import augment, add_ctrl
from algorithmic_sequential_problem import AlgorithmicSequentialProblem
from algorithmic_sequential_problem import DataTuple, AuxTuple


@AlgorithmicSequentialProblem.register
class ReadingSpan(AlgorithmicSequentialProblem):
    """
    # TODO : Documentation will be added soon
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
        """
        # TODO : Documentation will be added soon
        """

        # define control channel markers
        pos = [0, 0]
        ctrl_data = [0, 0]
        ctrl_dummy = [0, 1]
        ctrl_inter = [0, 1]
        # assign markers
        markers = ctrl_data, ctrl_dummy, pos

        # number sub sequences
        num_sub_seq = np.random.randint(self.num_subseq_min, self.num_subseq_max+1)

        # set the sequence length of each marker
        seq_length = np.random.randint(low=self.min_sequence_length, high=self.max_sequence_length + 1, size=num_sub_seq)

        #  generate subsequences for x and y
        x = [np.random.binomial(1, self.bias, (self.batch_size, n, self.data_bits)) for n in seq_length]
        x_last = [a[:, None, -1, :] for a in x]

        # create the target
        seq_length_tdummies = sum(seq_length) + seq_length.shape[0] + 1
        dummies_target = np.zeros([self.batch_size, seq_length_tdummies, self.data_bits], dtype=np.float32)
        targets = np.concatenate([dummies_target] + x_last, axis=1)

        # data of x and dummies
        xx = [augment(seq, markers, ctrl_start=[1,0], add_marker_data=True, add_marker_dummy = False) for seq in x]

        # data of x
        data_1 = [arr for a in xx for arr in a[:-1]]

        # this is a marker between sub sequence x and dummies
        inter_seq = add_ctrl(np.zeros((self.batch_size, 1, self.data_bits)), ctrl_inter, pos)

        # dummies of x
        x_dummy_last = [a[:, None, -1, :] for b in xx for a in b[-1:]]

        # concatenate all parts of the inputs
        inputs = np.concatenate(data_1 + [inter_seq] + x_dummy_last, axis=1)

        # PyTorch variables
        inputs = torch.from_numpy(inputs).type(self.dtype)
        targets = torch.from_numpy(targets).type(self.dtype)
        # TODO: batch might have different sequence lengths
        mask_all = inputs[..., 0:self.control_bits] == 1
        mask = mask_all[..., 0]
        for i in range(self.control_bits):
            mask = mask_all[..., i] * mask

        # TODO: fix the batch indexing
        # rest channel values of data dummies
        inputs[:, mask[0], 0:self.control_bits] = 0

        #return DataTuple(inputs, target, mask)
        data_tuple = DataTuple(inputs, targets)
        aux_tuple = AuxTuple(mask)

        return data_tuple, aux_tuple


    # method for changing the maximum length, used mainly during curriculum learning
    def set_max_length(self, max_length):
        self.max_sequence_length = max_length


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    params = {'control_bits': 2, 'data_bits': 8, 'batch_size': 2,
              'min_sequence_length': 1, 'max_sequence_length': 10, 
              'bias': 0.5, 'num_subseq_min':4 ,'num_subseq_max': 4}
    # Create problem object.
    problem = ReadingSpan(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    data_tuple,  aux_tuple = next(generator)
    # Display single sample (0) from batch.
    problem.show_sample(data_tuple, aux_tuple)

