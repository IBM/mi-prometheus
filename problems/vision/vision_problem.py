#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""algorithmic_sequential_problem.py: abstract base class for algorithmic, sequential problems"""
__author__      = "Tomasz Kornuta"

import abc
import collections
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

_DataTuple = collections.namedtuple('DataTuple', ('inputs', 'targets'))


class DataTuple(_DataTuple):
    """Tuple used by storing batches of data by data generators"""
    __slots__ = ()


_AuxTuple = collections.namedtuple('AuxTuple', ('mask'))


class AuxTuple(_AuxTuple):
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
        while True:
            yield self.generate_batch()

    def evaluate_loss_accuracy(self, logits, data_tuple, aux_tuple):

        self.criterion = nn.CrossEntropyLoss()

        # Unpack the data tuple.
        (_, targets) = data_tuple

        # 2. Calculate loss.
        # Compute loss using the provided criterion.
        inputs = logits[:, -1, :]
        loss = self.criterion(inputs, targets)

        # Calculate accuracy.
        pred = inputs.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct = pred.eq(targets.view_as(pred)).sum().item()

        batch_size = 64 # temporary
        accuracy = 100. * correct / batch_size

        return loss, accuracy

    def show_sample(self, inputs, targets):

        # show data.
        plt.xlabel('num_columns')
        plt.ylabel('num_rows')
        plt.title('number to be predicted:' + str(int(targets)))

        plt.imshow(inputs, interpolation='nearest', aspect='auto')
        # Plot!
        plt.show()
