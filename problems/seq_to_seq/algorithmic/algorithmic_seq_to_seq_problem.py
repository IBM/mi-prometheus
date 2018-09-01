#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""algorithmic_seq_to_seq_problem.py: abstract base class for algorithmic, sequential problems"""
__author__      = "Tomasz Kornuta, Younes Bouhadjar"

# Add path to main project directory - required for testing of the main function and see whether problem is working at all (!)
import os,  sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..','..','..','..')) 

import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from problems.problem import DataTuple
from problems.seq_to_seq.seq_to_seq_problem import SeqToSeqProblem

_AlgSeqAuxTuple = collections.namedtuple('AlgSeqAuxTuple', ('mask','seq_length','num_subsequences'))

class AlgSeqAuxTuple(_AlgSeqAuxTuple):
    """
    Tuple used by storing batches of data by algorithmic sequential problems.
    Contains three elements:

    - mask that might be used for evaluation of the loss function
    - length of sequence
    - number of subsequences

    """
    __slots__ = ()
    
class AlgorithmicSeqToSeqProblem(SeqToSeqProblem):
    ''' Base class for algorithmic, sequential problems. Provides some basic functionality usefull in all problems of such type'''

    def __init__(self, params):
        """ Initializes problem object. Calls base constructor. Sets nn.BCEWithLogitsLoss() as default loss function.
        
        :param params: Dictionary of parameters (read from configuration file). 
        """
        super(AlgorithmicSeqToSeqProblem, self).__init__(params)

        # Set default loss function - cross entropy.
        self.loss_function = nn.BCEWithLogitsLoss()

        # Extract "standard" list of parameters for algorithmic tasks.
        self.batch_size = params['batch_size']
        # Number of bits in one element.
        self.control_bits = params['control_bits']
        self.data_bits = params['data_bits']

        # Min and max lengts of a single subsequence (number of elements).
        self.min_sequence_length = params['min_sequence_length']
        self.max_sequence_length = params['max_sequence_length']

        # Add parameter denoting 0-1 distribution (DEFAULT: 0.5 i.e. equal).
        if 'bias' not in params:
            params.add_default_params({'bias': 0.5})
        self.bias = params['bias']

        # Set initial dtype.
        self.dtype = torch.FloatTensor     
          

    def calculate_accuracy(self, data_tuple, logits, aux_tuple):
        """ Calculate accuracy equal to mean difference between outputs and targets.
        WARNING: Applies mask (from aux_tuple) to both logits and targets!
        
        :param logits: Logits being output of the model.
        :param data_tuple: Data tuple containing inputs and targets.
        :param aux_tuple: Auxiliary tuple containing mask.        
        """

        # Check if mask should be is used - if so, apply.
        if (self.use_mask):
            masked_logits = logits[:, aux_tuple.mask[0], :]
            masked_targets = data_tuple.targets[:, aux_tuple.mask[0], :]
        else:
            masked_logits = logits
            masked_targets = data_tuple.targets            

        return (1 - torch.abs(torch.round(F.sigmoid(masked_logits)) - masked_targets)).mean()

    def turn_on_cuda(self, data_tuple, aux_tuple):
        """ Enables computations on GPU - copies all the matrices to GPU.
        This method has to be overwritten in derived class if one decides e.g. to add additional variables/matrices to aux_tuple.
        
        :param data_tuple: Data tuple.
        :param aux_tuple: Auxiliary tuple.
        :returns: Pair of Data and Auxiliary tupples with variables copied to GPU.
        """
        # Unpack tuples and copy data to GPU.
        gpu_inputs = data_tuple.inputs.cuda()
        gpu_targets = data_tuple.targets.cuda()
        gpu_mask = aux_tuple.mask.cuda()

        # Pack matrices to tuples.
        data_tuple = DataTuple(gpu_inputs, gpu_targets)

        # seq_length and num_subsequences are used only in logging, so are passed as they are i.e. stored in CPU.
        aux_tuple = AlgSeqAuxTuple(gpu_mask, aux_tuple.seq_length, aux_tuple.num_subsequences)

        return data_tuple, aux_tuple

    #def set_max_length(self, max_length):
    #    """ Sets maximum sequence lenth (property).
    #    
    #    :param max_length: Length to be saved as max.
    #    """
    #    self.max_sequence_length = max_length


    def add_ctrl(self, seq, ctrl, pos): 
        """
        Adds control channels to a sequence.
        """
        return np.insert(seq, pos, ctrl, axis=-1)


    def augment(self, seq, markers, ctrl_start=None, add_marker_data=False, add_marker_dummy=True):
        """
        Creates augmented sequence as well as end marker and a dummy sequence.
        """
        ctrl_data, ctrl_dummy, pos = markers

        w = self.add_ctrl(seq, ctrl_data, pos)
        start = self.add_ctrl(np.zeros((seq.shape[0], 1, seq.shape[2])), ctrl_start, pos)
        if add_marker_data:
            w = np.concatenate((start, w), axis=1)

        start_dummy = self.add_ctrl(np.zeros((seq.shape[0], 1, seq.shape[2])), ctrl_dummy, pos)
        ctrl_data_select = np.ones(len(ctrl_data))
        dummy = self.add_ctrl(np.zeros_like(seq), ctrl_data_select, pos)

        if add_marker_dummy:
            dummy = np.concatenate((start_dummy, dummy), axis=1)

        return [w, dummy]

    def add_statistics(self, stat_col):
        """
        Add accuracy, seq_length and num_subsequences statistics to collector. 

        :param stat_col: Statistics collector.
        """
        stat_col.add_statistic('acc', '{:12.10f}')
        stat_col.add_statistic('seq_length', '{:d}')
        #stat_col.add_statistic('num_subseq', '{:d}')
        stat_col.add_statistic('max_seq_length', '{:d}')


    def collect_statistics(self, stat_col, data_tuple, logits, aux_tuple):
        """
        Collects accuracy, seq_length and num_subsequences.

        :param stat_col: Statistics collector.
        :param data_tuple: Data tuple containing inputs and targets.
        :param logits: Logits being output of the model.
        :param aux_tuple: auxiliary tuple (aux_tuple) is not used in this function. 
        """
        stat_col['acc'] = self.calculate_accuracy(data_tuple, logits, aux_tuple)
        stat_col['seq_length'] = aux_tuple.seq_length
        #stat_col['num_subseq'] = aux_tuple.num_subsequences
        stat_col['max_seq_length'] = self.max_sequence_length


    def show_sample(self,  data_tuple, aux_tuple,  sample_number = 0):
        """ Shows the sample (both input and target sequences) using matplotlib.
            Elementary visualization.

        :param data_tuple: Data tuple.
        :param aux_tuple: Auxiliary tuple.
        :param sample_number: Number of sample in a batch (DEFAULT: 0)
        """
        
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        # Generate "canvas".
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1,  sharex=True,  sharey=False,
                                gridspec_kw={'width_ratios': [data_tuple.inputs.shape[1]],  'height_ratios':[10, 10,  1]})
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
        print("\ninputs:", data_tuple.inputs[sample_number, :, :])
        print("\ntargets:",data_tuple.targets[sample_number, :, :])
        print("\nmask:", aux_tuple.mask[sample_number:sample_number+1, :])
        print("\nseq_length:", aux_tuple.seq_length)
        print("\nnum_subsequences:", aux_tuple.num_subsequences)
        
        # show data.
        ax1.imshow(np.transpose(data_tuple.inputs[sample_number, :, :],  [1, 0]), interpolation='nearest', aspect='auto')        
        ax2.imshow(np.transpose(data_tuple.targets[sample_number, :, :],  [1, 0]), interpolation='nearest', aspect='auto')
        ax3.imshow(aux_tuple.mask[sample_number:sample_number+1, :],  interpolation='nearest', aspect='auto')  
        # Plot!
        plt.tight_layout()
        plt.show()


    def curriculum_learning_update_params(self, episode):
        """
        Updates problem parameters according to curriculum learning.
        In the case of algorithmic sequential problems it updates the max sequence length, depending on configuration parameters 

        :param episode: Number of the current episode.
        :returns: Boolean informing whether curriculum learning is finished (or wasn't active at all).
        """
        # Curriculum learning stop condition.
        curric_done = True
        try:
            # Read curriculum learning parameters.
            max_max_length = self.params['max_sequence_length']
            interval = self.curriculum_params['interval']
            initial_max_sequence_length = self.curriculum_params['initial_max_sequence_length']

            if self.curriculum_params['interval'] > 0:
                # Curriculum learning goes from the initial max length to the max length in steps of size 1
                max_length = initial_max_sequence_length + (episode // interval)
                if max_length >= max_max_length:
                    max_length = max_max_length
                else:
                    curric_done = False
                # Change max length.
                self.max_sequence_length = max_length
        except KeyError:
            pass
        # Return information whether we finished CL (i.e. reached max sequence length).
        return curric_done