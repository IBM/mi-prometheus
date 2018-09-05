__author__ = "Ryan McAvoy"

import torch
import numpy as np
from problems.problem import DataTuple
from problems.seq_to_seq.algorithmic.algorithmic_seq_to_seq_problem import AlgorithmicSeqToSeqProblem, AlgSeqAuxTuple
from misc.param_interface import ParamInterface


class SequenceEqualityCommandLines(AlgorithmicSeqToSeqProblem):
    """
    Class generating sequences of random bit-patterns and targets forcing the system to learn scratch pad problem (overwrite the memory).

    @Ryan: ARE YOU SURE? FIX THE CLASS DESCRIPTION!
    """

    def __init__(self, params):
        """
        Constructor - stores parameters. Calls parent class initialization.

        :param params: Dictionary of parameters.
        """
        # Call parent constructor - sets e.g. the loss function, dtype.
        # Additionally it extracts "standard" list of parameters for
        # algorithmic tasks, like batch_size, numbers of bits, sequences etc.
        super(SequenceEqualityCommandLines, self).__init__(params)

        assert self.control_bits >= 3, "Problem requires at least 3 control bits (currently %r)" % self.control_bits
        assert self.data_bits >= 1, "Problem requires at least 1 data bit (currently %r)" % self.data_bits

        # The bit that idicates whether we want to return true when sequences
        # are equal or not equal.
        self.predict_inverse = params.get('predict_inverse', True)

    def generate_batch(self):
        """Generates a batch  of size [BATCH_SIZE, SEQ_LENGTH, CONTROL_BITS+DATA_BITS].
        SEQ_LENGTH depends on number of sub-sequences and its lengths

        :returns: Tuple consisting of: input, output and mask
                  pattern of inputs: x1, x2, ...xn d
                  pattern of target: d, d,   ...d xn
                  mask: used to mask the data part of the target
                  xi, d: sub sequences, dummies

        """
        # define control channel markers
        # pos = [0, 0, 0]
        pos = np.zeros(self.control_bits)  # [0, 0, 0]

        # ctrl_data = [0, 0, 0]
        ctrl_data = np.zeros(self.control_bits)  # [0, 0, 0]

        # ctrl_dummy = [0, 0, 0 ]
        ctrl_dummy = np.zeros(self.control_bits)

        # ctrl_inter = [0, 1, 0]
        ctrl_inter = np.zeros(self.control_bits)
        ctrl_inter[1] = 1  # [0, 1, 0]

        # ctrl_y = [0, 0, 1]
        ctrl_y = np.zeros(self.control_bits)
        ctrl_y[2] = 1  # [0, 1, 0]

        # ctrl_start = [1, 0, 0]
        ctrl_start = np.zeros(self.control_bits)
        ctrl_start[0] = 1  # [1, 0, 0]

        # assign markers
        markers = ctrl_data, ctrl_dummy, pos

        # set the sequence length of each marker
        seq_length = np.random.randint(
            low=self.min_sequence_length, high=self.max_sequence_length + 1)

        #  generate subsequences for x and y
        x = [np.array(np.random.binomial(
            1, self.bias, (self.batch_size, seq_length, self.data_bits)))]

        # Generate the second sequence which is either a scrambled version of the first
        # or exactly identical with approximately 50% probability (technically the scrambling
        # allows them to be the same with a very low chance)

        # First generate a random binomial of the same size as x, this will be
        # used be used with an xor operation to scamble x to get y
        xor_scrambler = np.array(np.random.binomial(1, self.bias, x[0].shape))

        # Create a mask that will set entire batches of the xor_scrambler to zero. The batches that are zero
        # will force the xor to return the original x for that batch
        scrambler_mask = np.array(np.random.binomial(
            1, self.bias, (self.batch_size,)))
        xor_scrambler = np.array(
            xor_scrambler * scrambler_mask[:, np.newaxis, np.newaxis])

        aux_seq = np.array(np.logical_xor(x[0], xor_scrambler))

        # if the xor scambler is all zeros then x and y will be the same so
        # target will be true
        actual_target = np.array(np.any(xor_scrambler, axis=(1, 2)))

        if self.predict_inverse:
            # if the xor scambler is all zeros then x and y will be the same so
            # target will be true
            actual_target = actual_target[:, np.newaxis, np.newaxis]
        else:
            actual_target = np.logical_not(
                actual_target[:, np.newaxis, np.newaxis])

        # create the target
        seq_length_tdummies = 2 * seq_length + 1
        dummies_target = np.zeros(
            [self.batch_size, seq_length_tdummies, 1], dtype=np.float32)
        target = np.concatenate((dummies_target, actual_target), axis=1)

        # data of x and dummies
        xx = [
            self.augment(
                seq,
                markers,
                ctrl_start=ctrl_start,
                add_marker_data=True,
                add_marker_dummy=False) for seq in x]

        # data of x
        data_1 = [arr for a in xx for arr in a[:-1]]

        # this is a marker between sub sequence x and dummies
        inter_seq = [self.add_ctrl(
            np.zeros((self.batch_size, 1, self.data_bits)), ctrl_inter, pos)]
        # Second Sequence for comparison

        markers2 = ctrl_y, ctrl_dummy, pos
        yy = [self.augment(aux_seq, markers2, ctrl_start=ctrl_y,
                           add_marker_data=False, add_marker_dummy=False)]
        data_2 = [arr for a in yy for arr in a[:-1]]
        data_2[0][:, -1, 0:self.control_bits] = np.ones(len(ctrl_dummy))

        recall_seq = [self.add_ctrl(
            np.zeros((self.batch_size, 1, self.data_bits)), ctrl_dummy, pos)]
        dummy_data = [
            self.add_ctrl(
                np.zeros(
                    (self.batch_size, 1, self.data_bits)), np.ones(
                    len(ctrl_dummy)), pos)]

        inputs = np.concatenate(data_1 + inter_seq + data_2, axis=1)

        # PyTorch variables
        inputs = torch.from_numpy(inputs).type(self.dtype)
        target = torch.from_numpy(target).type(self.dtype)
        # TODO: batch might have different sequence lengths
        mask_all = inputs[..., 0:self.control_bits] == 1
        mask = mask_all[..., 0]
        for i in range(self.control_bits):
            mask = mask_all[..., i] * mask

        # TODO: fix the batch indexing
        # rest channel values of data dummies
        inputs[:, mask[0], 0:self.control_bits] = torch.tensor(
            ctrl_y).type(self.dtype)

        # Return tuples.
        data_tuple = DataTuple(inputs, target)
        aux_tuple = AlgSeqAuxTuple(mask, seq_length, 1)

        return data_tuple, aux_tuple

    # method for changing the maximum length, used mainly during curriculum
    # learning
    def set_max_length(self, max_length):
        self.max_sequence_length = max_length


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    params = ParamInterface()
    params.add_custom_params({'control_bits': 4, 'data_bits': 8, 'batch_size': 1,
                              # 'predict_inverse': False,
                              'min_sequence_length': 1, 'max_sequence_length': 2})
    # Create problem object.
    problem = SequenceEqualityCommandLines(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    data_tuple, aux_tuple = next(generator)

    # Display single sample (0) from batch.
    problem.show_sample(data_tuple, aux_tuple)
