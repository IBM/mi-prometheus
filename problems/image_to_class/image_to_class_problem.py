#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""algorithmic_sequential_problem.py: abstract base class for vision problems"""
__author__      = "Tomasz Kornuta"

from problems.problem import Problem
from problems.problem import DataTuple
import torch.nn as nn


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


    def show_sample(self, inputs, targets):
        import matplotlib.pyplot as plt

        # show data.
        plt.xlabel('num_columns')
        plt.ylabel('num_rows')
        plt.title('Target class: ' + str(int(targets[0])))

        plt.imshow(inputs, interpolation='nearest', aspect='auto')
        # Plot!
        plt.show()
