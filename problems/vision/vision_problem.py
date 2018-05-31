#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""algorithmic_sequential_problem.py: abstract base class for algorithmic, sequential problems"""
__author__      = "Tomasz Kornuta"

import abc
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rc


_DataTuple = collections.namedtuple('DataTuple', ('inputs', 'targets'))

class DataTuple(_DataTuple):
    """Tuple used by storing batches of data by data generators"""
    __slots__ = ()
    
class VisionProblem(metaclass=abc.ABCMeta):
    ''' Abstract base class for algorithmic, sequential problems. Provides some basic functionality usefull in all problems of such type'''

    @abc.abstractmethod
    def generate_batch(self):
        """Generates batch of sequences of given length. """

    def return_generator(self):
        """Returns a generator yielding a batch  of size [BATCH_SIZE, 2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS].
        Additional elements of sequence are  start and stop control markers, stored in additional bits.
       
        : returns: A tuple: input with shape [BATCH_SIZE, 2*SEQ_LENGTH+2, CONTROL_BITS+DATA_BITS], output 
        """
        # Create "generator".
        return self.generate_batch()


    def show_sample(self, inputs, targets, sample_number = 0):

        # show data.
        plt.xlabel('num_columns')
        plt.ylabel('num_rows')
        plt.title('number to be predicted:' + str(int(targets)))

        plt.imshow(inputs[sample_number, sample_number, :, :], interpolation='nearest', aspect='auto')
        # Plot!
        plt.show()
