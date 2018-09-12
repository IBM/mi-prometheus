#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""image_text_to_class_problem.py: contains abstract base class for VQA problems"""
__author__ = "Tomasz Kornuta & Vincent Marois"


import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
from problems.problem import Problem, DataDict


class ObjectRepresentation:
    """
    Class storing features of the object being present in a given scene.

    Used in ShapeColorQuery and SortOfCLEVR.

    """

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
        Initializes problem, calls base class initialization. Set loss function
        to CrossEntropy.

        :param params: Dictionary of parameters (read from configuration file).

        """
        # Call base class constructors.
        super(ImageTextToClassProblem, self).__init__(params)

        # set default loss function
        self.loss_function = nn.CrossEntropyLoss()

        # set default data_definitions dict
        self.data_definitions = {'text': {'type': int},
                                'images': {'width': 256, 'type': torch.Tensor},
                                'targets': {'size': 1, 'type': int}}

    def calculate_accuracy(self, data_dict, logits):
        """
        Calculates the accuracy as the mean number of correct answers in a given batch.

        :param data_dict: DataDict containing inputs and targets.

        :param logits: Predictions being output of the model.
        """

        # Get the index of the max log-probability.
        pred = logits.max(1, keepdim=True)[1]
        correct = pred.eq(data_dict['targets'].view_as(pred)).sum().item()

        # Calculate the accuracy.
        batch_size = logits.size(0)
        accuracy = correct / batch_size

        return accuracy

    def __getitem__(self, item):
        """
        Getter that returns an individual sample from the problem's associated dataset (that can be generated \
        on-the-fly, or retrieved from disk. It can also possibly be composed of several files.).

        To be redefined in subclasses.

        **The getter should return a DataDict: its keys should be defined by `self.data_definitions` keys.**

        e.g.:
            >>> data_dict = DataDict({key: None for key in self.data_definitions.keys()})
            >>> # you can now access each value by its key and assign the corresponding object (e.g. `torch.Tensor` etc)
            >>> ...
            >>> return data_dict



        .. warning::

            In a future version of `mi-prometheus`, multiprocessing will be supported for data loading.
            To construct a batch (say 64 samples), the indexes will be distributed among several workers (say 4, so that
            each worker has 16 samples to retrieve). It is best that samples can be accessed individually in the dataset
            folder so that there is no mutual exclusion between the workers and the performance is not degraded.

        :param index: index of the sample to return.

        :return: DataDict containing the sample.

        """

        return DataDict({key: None for key in self.data_definitions.keys()})

    def collate_fn(self, batch):
        """
        Generates a batch of samples from a list of individuals samples retrieved by `__getitem__`.
        The default collate_fn is torch.utils.data.default_collate.

        .. note::
            **Simply returning self.collate_fn(batch) for now. It is encouraged to redefine it in the subclasses.**


        :param batch: Should be a list of DataDict retrieved by `__getitem__`, each containing tensors, numbers,
        dicts or lists.

        :return: DataDict containing the created batch.

        """
        return default_collate

    def add_statistics(self, stat_col):
        """
        Add accuracy statistic to collector.

        :param stat_col: Statistics collector.

        """
        stat_col.add_statistic('acc', '{:12.10f}')

    def collect_statistics(self, stat_col, data_dict, logits):
        """
        Collects accuracy.

        :param stat_col: Statistics collector.

        :param data_dict: DataDict containing inputs and targets.

        :param logits: Predictions being output of the model.

        """
        stat_col['acc'] = self.calculate_accuracy(data_dict, logits)


if __name__ == '__main__':

    sample = ImageTextToClassProblem(params={})[0]
    # equivalent to ImageTextToClassProblem(params={}).__getitem__(index=0)

    print(repr(sample))