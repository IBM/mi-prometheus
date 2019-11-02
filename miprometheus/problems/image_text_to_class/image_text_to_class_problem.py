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

"""image_text_to_class_problem.py: contains abstract base class for Visual Question Answering problems."""
__author__ = "Tomasz Kornuta & Vincent Marois"


import torch
import torch.nn as nn
from miprometheus.problems.problem import Problem


class ObjectRepresentation(object):
    """
    Class storing some features representing an object being present in a given scene.

    Used in ShapeColorQuery and SortOfCLEVR.

    """

    def __init__(self, x, y, color, shape):
        """
        Represents an object.

        :param x: x coordinate.
        :param y: y coordinate.
        :param color: Color of the object.
        :param shape: Shape of the object.

        """
        self.x = x
        self.y = y
        self.color = color
        self.shape = shape


class ImageTextToClassProblem(Problem):
    """
    Abstract base class for VQA (`Visual Question Answering`) problems.

    Problem classes like CLEVR inherits from it.

    Provides some basic features useful in all problems of such type.

    """
    def __init__(self, params):
        """
        Initializes problem:

            - Calls ``problems.problem.Problem`` class constructor,
            - Sets loss function to ``CrossEntropy``,

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).

        """
        # Call base class constructors.
        super(ImageTextToClassProblem, self).__init__(params)
        # "Default" problem name.
        self.name = 'ImageTextToClassProblem'

        # set default loss function
        self.loss_function = nn.CrossEntropyLoss()

    def evaluate_loss(self, data_dict, logits):
        """
        Calculates loss between the predictions / logits and targets (from ``data_dict``) using the selected \
        loss function.

        :param data_dict: DataDict containing (among others) inputs and targets.
        :type data_dict: :py:class:`miprometheus.utils.DataDict`

        :param logits: Predictions of the model.

        :return: Loss.
        """

        # Compute loss using the provided loss function.
        loss = self.loss_function(logits, data_dict['targets'].type(self.app_state.LongTensor))

        return loss

    def calculate_accuracy(self, data_dict, logits):
        """
        Calculates the accuracy as the mean number of correct answers in a given batch.

        :param data_dict: DataDict containing the targets.
        :type data_dict: DataDict

        :param logits: Predictions of the model.

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
        # Add basic statistics.
        super(ImageTextToClassProblem, self).add_statistics(stat_col)
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
        # Collect basic statistics.
        super(ImageTextToClassProblem, self).collect_statistics(stat_col, data_dict, logits)
        stat_col['acc'] = self.calculate_accuracy(data_dict, logits)
        stat_col['batch_size'] = logits.shape[0] # Batch major.

    def add_aggregators(self, stat_agg):
        """
        Adds problem-dependent statistical aggregators to ``StatisticsAggregator``.

        :param stat_agg: ``StatisticsAggregator``.

        """
        # Add basic aggregators.
        super(ImageTextToClassProblem, self).add_aggregators(stat_agg)

        stat_agg.add_aggregator('acc', '{:12.10f}')  # represents the average accuracy
        stat_agg.add_aggregator('acc_min', '{:12.10f}')
        stat_agg.add_aggregator('acc_max', '{:12.10f}')
        stat_agg.add_aggregator('acc_std', '{:12.10f}')
        stat_agg.add_aggregator('samples_aggregated', '{:06d}')

    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates the statistics collected by ``StatisticsCollector`` and adds the results to ``StatisticsAggregator``.

        :param stat_col: ``StatisticsCollector``.

        :param stat_agg: ``StatisticsAggregator``.

        """
        # Aggregate base statistics.
        super(ImageTextToClassProblem, self).aggregate_statistics(stat_col, stat_agg)

        stat_agg['acc_min'] = min(stat_col['acc'])
        stat_agg['acc_max'] = max(stat_col['acc'])
        stat_agg['acc'] = torch.mean(torch.tensor(stat_col['acc']))
        stat_agg['acc_std'] = 0.0 if len(stat_col['acc']) <= 1 else torch.std(torch.tensor(stat_col['acc']))
        stat_agg['samples_aggregated'] = sum(stat_col['batch_size'])


if __name__ == '__main__':

    from miprometheus.utils.param_interface import ParamInterface

    sample = ImageTextToClassProblem(ParamInterface())[0]
    # equivalent to ImageTextToClassProblem(params={}).__getitem__(index=0)

    print(repr(sample))
