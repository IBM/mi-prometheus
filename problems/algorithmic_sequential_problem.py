#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""algorithmic_sequential_problem.py: abstract base class for algorithmic, sequential problems"""
__author__      = "Tomasz Kornuta"

import abc
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import rc


_DataTuple = collections.namedtuple('DataTuple', ('inputs', 'targets',  'mask'))

class DataTuple(_DataTuple):
    """Tuple used by storing batches of data by data generators"""
    __slots__ = ()
    
class AlgorithmicSequentialProblem(metaclass=abc.ABCMeta):
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
            # Yield batch.
            yield self.generate_batch()

    def evaluate_loss_accuracy(self, logits, data_tuple, use_mask):
        self.criterion = nn.BCEWithLogitsLoss()
        # Unpack the data tuple.
        (inputs, targets, mask) = data_tuple

        # 2. Calculate loss.
        # Check if mask should be is used - if so, apply.
        if use_mask:
            masked_logits = logits[:, mask[0], :]
            masked_targets = targets[:, mask[0], :]
        else:
            masked_logits = logits
            masked_targets = targets

        # Compute loss using the provided criterion.
        loss = self.criterion(masked_logits, masked_targets)

        # Calculate accuracy.
        accuracy = (1 - torch.abs(torch.round(F.sigmoid(masked_logits)) - masked_targets)).mean()

        return loss, accuracy

    def set_max_length(self, max_length):
        self.max_sequence_length = max_length

    def show_sample(self,  inputs,  targets, mask,  sample_number = 0):
        """ Shows the sample (both input and target sequences) using matplotlib."""
                # Change fonts globally - for all axes at once.
        rc('font',**{'family':'Times New Roman'})
        
        # Generate "canvas".
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1,  sharex=True,  sharey=False,
                                gridspec_kw={'width_ratios': [inputs.shape[1]],  'height_ratios':[10, 10,  1]})
        # Set ticks.
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax3.yaxis.set_major_locator(ticker.NullLocator())

        # Set labels.
        ax1.set_title('Inputs') 
        ax1.set_ylabel('Control/Data bits')
        ax2.set_title('Targets')
        ax2.set_ylabel('Data bits')
        ax3.set_title('Target mask')
        ax3.set_ylabel('Mask bit')
        ax3.set_xlabel('Item number', fontname='Times New Roman', fontsize=13)

        # print data
        print("\ninputs:", inputs[sample_number, :, :])
        print("\ntargets:",targets[sample_number, :, :])
        print("\nmask:", mask[sample_number:sample_number+1, :])
        
        # show data.
        ax1.imshow(np.transpose(inputs[sample_number, :, :],  [1, 0]), interpolation='nearest', aspect='auto')        
        ax2.imshow(np.transpose(targets[sample_number, :, :],  [1, 0]), interpolation='nearest', aspect='auto')
        ax3.imshow(mask[sample_number:sample_number+1, :],  interpolation='nearest', aspect='auto')  
        # Plot!
        plt.tight_layout()
        plt.show()
