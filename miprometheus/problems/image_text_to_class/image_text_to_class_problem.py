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
            - sets ``self.data_definitions`` to:

                >>>         self.data_definitions = {'texts': {'size': [-1, -1], 'type': [torch.Tensor]},
                >>>                                  'images': {'size': [-1, -1, -1, 3], 'type': [torch.Tensor]},
                >>>                                  'targets': {'size': [-1, 1], 'type': [torch.Tensor]}
                >>>                                 }

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).

        """
        # Call base class constructors.
        super(ImageTextToClassProblem, self).__init__(params)

        # set default loss function
        self.loss_function = nn.CrossEntropyLoss()

        # set default data_definitions dict
        self.data_definitions = {'texts': {'size': [-1, -1], 'type': [torch.Tensor]},
                                 'images': {'size': [-1, -1, -1, 3], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1, 1], 'type': [torch.Tensor]}
                                 }

        # "Default" problem name.
        self.name = 'ImageTextToClassProblem'

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
        stat_col.add_statistic('acc', '{:12.10f}')

    def collect_statistics(self, stat_col, data_dict, logits):
        """
        Collects accuracy.

        :param stat_col: ``StatisticsCollector``.

        :param data_dict: DataDict containing the targets and the mask.
        :type data_dict: DataDict

        :param logits: Predictions of the model.

        """
        stat_col['acc'] = self.calculate_accuracy(data_dict, logits)


if __name__ == '__main__':

    from miprometheus.utils.param_interface import ParamInterface

    sample = ImageTextToClassProblem(ParamInterface())[0]
    # equivalent to ImageTextToClassProblem(params={}).__getitem__(index=0)

    print(repr(sample))