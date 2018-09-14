#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""image_to_class_problem.py: contains base class for image classification problems"""
__author__ = "Younes Bouhadjar, Vincent Marois"
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data.dataloader import default_collate

from problems.problem import Problem, DataDict


class ImageToClassProblem(Problem):
    """
    Abstract base class for image classification problems.

    TODO: DOCUMENTATION

    Provides some basic functionality useful in all problems of such
    type.

    """

    def __init__(self, params):
        """
        Initializes problem, calls base class initialization. Set loss function
        to CrossEntropy.

        :param params: Dictionary of parameters (read from configuration file).

        """
        # Call base class constructors.
        super(ImageToClassProblem, self).__init__(params)

        # set default loss function
        self.loss_function = nn.CrossEntropyLoss()

        # set default data_definitions dict
        self.data_definitions = {'images': {'size': [-1, 3, -1, -1], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1, 1], 'type': [torch.Tensor]},
                                 'targets_label': {'size': [-1, 1], 'type': [list, str]}
                                 }

    def calculate_accuracy(self, data_dict, logits):
        """
        Calculates accuracy equal to mean number of correct classification in a given batch.

        :param logits: Logits being output of the model.

        :param data_dict: DataDict containing inputs and targets.

        :return loss

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
        return default_collate(batch)

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

        :param logits: Predictions being output of the model.

        """
        stat_col['acc'] = self.calculate_accuracy(data_dict, logits)

    def show_sample(self, data_dict, sample_number=0):
        """
        Shows a sample from the batch.

        :param data_dict: Tuple containing inputs and targets.

        :param sample_number: Number of sample in batch (DEFAULT: 0)

        """
        import matplotlib.pyplot as plt

        # Unpack tuples.
        images, targets, labels = data_dict.values()

        # Get sample.
        image = images[sample_number].cpu().detach().numpy()
        target = targets[sample_number].cpu().detach().numpy()
        label = labels[sample_number]

        # Reshape image.
        if image.shape[0] == 1:
            # This is a single channel image - get rid of that dimension
            image = np.squeeze(image, axis=0)
        else:
            # More channels - move channels to axis2, according to matplotilb doc it should be ok
            # (X : array_like, shape (n, m) or (n, m, 3) or (n, m, 4))
            image = image.transpose(1, 2, 0)

        # Show data.
        plt.title('Target class: {} ({})'.format(label, target))
        plt.imshow(image, interpolation='nearest', aspect='auto')

        # Plot!
        plt.show()


if __name__ == '__main__':

    from utils.param_interface import ParamInterface

    sample = ImageToClassProblem(ParamInterface())[0]
    # equivalent to ImageToClassProblem(params={}).__getitem__(index=0)

    print(repr(sample))
