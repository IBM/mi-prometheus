#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""image_to_class_problem.py: contains abstract base class for image classification problems"""
__author__      = "Tomasz Kornuta"

import numpy as np
import torch.nn as nn

from problems.problem import Problem
from problems.problem import DataTuple


class ImageToClassProblem(Problem):
    ''' Abstract base class for image classification problems. Provides some basic functionality usefull in all problems of such type'''


    def __init__(self, params):
        """ 
        Initializes problem, calls base class initialization. Set loss function to CrossEntropy.

        :param params: Dictionary of parameters (read from configuration file).        
        """ 
        # Call base class constructors.
        super(ImageToClassProblem, self).__init__(params)

        self.loss_function = nn.CrossEntropyLoss()

    def calculate_accuracy(self, data_tuple, logits, _):
        """ Calculates accuracy equal to mean number of correct classification in a given batch.
        WARNING: Applies mask (from aux_tuple) to logits!
        
        :param logits: Logits being output of the model.
        :param data_tuple: Data tuple containing inputs and targets.
        :param _: auxiliary tuple (aux_tuple) is not used in this function. 
        """

        # Get the index of the max log-probability.
        pred = logits.max(1, keepdim=True)[1]  
        correct = pred.eq(data_tuple.targets.view_as(pred)).sum().item()

        # Calculate the accuracy.
        batch_size = logits.size(0)
        accuracy = correct / batch_size

        return accuracy

    def add_statistics(self, stat_col):
        """
        Add accuracy statistic to collector. 

        :param stat_col: Statistics collector.
        """
        stat_col.add_statistic('acc', '{:12.10f}')
    

    def collect_statistics(self, stat_col, data_tuple, logits, _):
        """
        Collects accuracy.

        :param stat_col: Statistics collector.
        :param data_tuple: Data tuple containing inputs and targets.
        :param logits: Logits being output of the model.
        :param _: auxiliary tuple (aux_tuple) is not used in this function. 
        """
        stat_col['acc'] = self.calculate_accuracy(data_tuple, logits, _)


    def show_sample(self, data_tuple, aux_tuple, sample_number = 0):
        """ 
        Shows a sample from the batch.

        :param data_tuple: Tuple containing inputs and targets.
        :param aux_tuple: Auxiliary tuple containing scene descriptions.
        :param sample_number: Number of sample in batch (DEFAULT: 0) 
        """
        import matplotlib.pyplot as plt

        # Unpack tuples.
        images, targets = data_tuple

        # Get sample.
        image = images[sample_number].cpu().detach().numpy()
        target = targets[sample_number].cpu().detach().numpy()
        label = aux_tuple.label[sample_number]

        # Reshape image.
        if (image.shape[0] == 1):
            # This is single channel image - get rid of that dimension
            image = np.squeeze(image, axis=0)
        else:
            # More channels - move channels to axis2, according to matplotilb doc it should be ok 
            # (X : array_like, shape (n, m) or (n, m, 3) or (n, m, 4))    
            image = image.transpose(1, 2, 0)

        # Show data.
        plt.title('Target class: {} ({})'.format(label, target) )
        plt.imshow(image, interpolation='nearest', aspect='auto')

        # Plot!
        plt.show()
