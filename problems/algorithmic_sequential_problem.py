#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""algorithmic_sequential_problem.py: abstract base class for algorithmic, sequential problems"""
__author__      = "Tomasz Kornuta"

import abc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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


    def show_sample(self,  inputs,  targets, mask,  sample_number = 0):
        """ Shows the sample (both input and target sequences) using matplotlib."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3,  gridspec_kw = {'width_ratios':[1,  1,  1]},  sharex=True)
        # Set ticks.
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax3.yaxis.set_major_locator(ticker.NullLocator())

        # Set labels.
        ax1.set_title('Inputs') 
        ax1.set_ylabel('Control/Data bits')     
        ax1.set_xlabel('Item number')   
        ax2.set_title('Targets')
        #ax2.set_ylabel('Data bits')     
        ax2.set_xlabel('Item number')   
        ax3.set_title('Target mask')
        ax3.set_xlabel('Item number')   
        
        # Set data.
        ax1.imshow(np.transpose(inputs[sample_number, :, :],  [1, 0]))        
        ax2.imshow(np.transpose(targets[sample_number, :, :],  [1, 0]))
        ax3.imshow(mask[sample_number:sample_number+1, :])  
        # Plot!
        plt.show()
