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

"""image_to_class_problem.py: contains base class for image classification problems."""
__author__ = "Younes Bouhadjar & Vincent Marois"
import torch
import numpy as np
import torch.nn as nn

from problems.problem import Problem, DataDict


class ImageToClassProblem(Problem):
    """
    Abstract base class for image classification problems.

    Problem classes like MNIST & CIFAR10 inherits from it.

    Provides some basic features useful in all problems of such type.

    """

    def __init__(self, params):
        """
        Initializes problem:

            - Calls ``problems.problem.Problem`` class constructor,
            - Sets loss function to ``CrossEntropy``,
            - sets ``self.data_definitions`` to:

                >>> self.data_definitions = {'images': {'size': [-1, 3, -1, -1], 'type': [torch.Tensor]},
                >>>                          'targets': {'size': [-1, 1], 'type': [torch.Tensor]},
                >>>                          'targets_label': {'size': [-1, 1], 'type': [list, str]}
                >>>                         }

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).

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

        # "Default" problem name.
        self.name = 'ImageToClassProblem'

    def calculate_accuracy(self, data_dict, logits):
        """
        Calculates accuracy equal to mean number of correct classification in a given batch.

        :param logits: Predictions of the model.

        :param data_dict: DataDict containing the targets.
        :type data_dict: DataDict

        :return: Accuracy.

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
        Add accuracy statistic to ``StatisticsCollector``.

        :param stat_col: ``StatisticsCollector``.

        """
        stat_col.add_statistic('acc', '{:12.10f}')
        stat_col.add_statistic('batch_size', '{:06d}')

    def collect_statistics(self, stat_col, data_dict, logits):
        """
        Collects accuracy.

        :param stat_col: ``StatisticsCollector``.

        :param data_dict: DataDict containing the targets and the mask.
        :type data_dict: DataDict

        :param logits: Predictions of the model.

        """
        stat_col['acc'] = self.calculate_accuracy(data_dict, logits)
        stat_col['batch_size'] = logits.shape[0] # Batch major.

    def add_aggregators(self, stat_agg):
        """
        Adds problem-dependent statistical aggregators to ``StatisticsAggregator``.

        :param stat_agg: ``StatisticsAggregator``.

        """
        stat_agg.add_aggregator('acc', '{:12.10f}')  # represents the average accuracy
        stat_agg.add_aggregator('acc_min', '{:12.10f}')
        stat_agg.add_aggregator('acc_max', '{:12.10f}')
        stat_agg.add_aggregator('acc_std', '{:12.10f}')
        stat_agg.add_aggregator('samples_aggregated', '{:006d}')  # represents the average accuracy


    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates the statistics collected by ''StatisticsCollector'' and adds the results to ''StatisticsAggregator''.

        :param stat_col: ``StatisticsCollector``.

        :param stat_agg: ``StatisticsAggregator``.

        """
        stat_agg['acc_min'] = min(stat_col['acc'])
        stat_agg['acc_max'] = max(stat_col['acc'])
        stat_agg['acc'] = torch.mean(torch.tensor(stat_col['acc']))
        stat_agg['acc_std'] = torch.std(torch.tensor(stat_col['acc']))
        stat_agg['samples_aggregated'] = sum(stat_col['batch_size'])


    def show_sample(self, data_dict, sample_number=0):
        """
        Shows a sample from the batch.

        :param data_dict: ``DataDict`` containing inputs and targets.
        :type data_dict: DataDict

        :param sample_number: Number of sample in batch (default: 0)
        :type sample_number: int

        """
        import matplotlib.pyplot as plt

        # Unpack dict.
        images, targets, labels = data_dict.values()

        # Get sample.
        image = images[sample_number].cpu().detach().numpy()
        target = targets[sample_number].cpu().detach().numpy()
        label = labels[sample_number]

        # Reshape image.
        if image.shape[0] == 1:
            # This is a single channel image - get rid of this dimension
            image = np.squeeze(image, axis=0)
        else:
            # More channels - move channels to axis2, according to matplotilb documentation.
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
