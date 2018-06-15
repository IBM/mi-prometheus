#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""problem.py: contains base class for all seq2seq problems"""
__author__      = "Tomasz Kornuta"

import collections
import torch.nn as nn
from problems.problem import Problem

class SeqToSeqProblem(Problem):
    """ Class representing base class for all sequential problems.
    """
    
    def __init__(self, params):
        """ Initializes problem object. Calls base constructor.
        
        :param params: Dictionary of parameters (read from configuration file). 
        """
        super(SeqToSeqProblem, self).__init__(params)

    def evaluate_loss(self, data_tuple, logits, aux_tuple):
        """ Calculates accuracy equal to mean number of correct predictions in a given batch.
        WARNING: Applies mask (from aux_tuple) to both logits and targets!
        
        :param logits: Logits being output of the model.
        :param data_tuple: Data tuple containing inputs and targets.
        :param aux_tuple: Auxiliary tuple containing mask.
        """
        # Check if mask should be is used - if so, apply. TODO!
        masked_logits = logits[:, aux_tuple.mask[0], :]
        masked_targets = data_tuple.targets[:, aux_tuple.mask[0], :]

        # Compute loss using the provided loss function between predictions/logits and targets!
        loss = self.loss_function(masked_logits, masked_targets)

        return loss