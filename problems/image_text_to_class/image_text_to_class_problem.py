#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""image_text_to_class_problem.py: contains abstract base class for VQA problems"""
__author__ = "Tomasz Kornuta & Vincent Marois"

import torch.nn as nn
from problems.problem import Problem, DataDict


class ImageTextDict(DataDict):
    """
    Dict used for storing batches of image-text pairs by e.g. VQA problems
    """
    def __init__(self, *args, **kwargs):
        super(ImageTextDict, self).__init__(*args, **kwargs)

        del self['inputs']
        self['images'] = None
        self['texts'] = None
        self['scene_descriptions'] = None

        self.__dict__.update(*args, **kwargs)

    def __repr__(self):
        """
        Echoes class, id, & reproducible representation in the Read–Eval–Print Loop.
        """
        return '{}, ImageTextDict({})'.format(super(DataDict, self).__repr__(), self.__dict__)


class ObjectRepresentation:
    """ Class storing features of the object being present in a given scene. """
    def __init__(self, x, y, color, shape):
        self.x = x
        self.y = y
        self.color = color
        self.shape = shape


class ImageTextToClassProblem(Problem):
    """
    Abstract base class for VQA  (Visual Question Answering) problems.
    Provides some basic functionality useful in all problems of such type.
    """
    def __init__(self, params):
        """ 
        Initializes problem, calls base class initialization. Set loss function to CrossEntropy.

        :param params: Dictionary of parameters (read from configuration file).        
        """ 
        # Call base class constructors.
        super(ImageTextToClassProblem, self).__init__(params)

        self.loss_function = nn.CrossEntropyLoss()

    def calculate_accuracy(self, data_dict, logits, _):
        """ Calculates accuracy equal to mean number of correct answers in a given batch.

        :param data_dict: DataDict containing inputs and targets.
        :param logits: Logits being output of the model.
        :param _: auxiliary tuple (aux_tuple) is not used in this function.
        """

        # Get the index of the max log-probability.
        pred = logits.max(1, keepdim=True)[1]
        correct = pred.eq(data_dict['targets'].view_as(pred)).sum().item()

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

    def collect_statistics(self, stat_col, data_dict, logits, _):
        """
        Collects accuracy.

        :param stat_col: Statistics collector.
        :param data_dict: DataDict containing inputs and targets.
        :param logits: Logits being output of the model.
        :param _: auxiliary tuple (aux_tuple) is not used in this function. 
        """
        stat_col['acc'] = self.calculate_accuracy(data_dict, logits, _)


if __name__=='__main__':

    imagetextdict= ImageTextDict()
    print(imagetextdict.numpy())