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
"""video_to_class_problem.py: abstract base class for sequential vision problems"""
__author__ = "Tomasz Kornuta, Younes Bouhadjar"

import torch.nn as nn
from problems.problem import Problem


class VideoToClassProblem(Problem):
    """
    Base class for vision classification problems.

    Provides some basic functionality usefull in all problems of such
    type

    """

    def __init__(self, params):
        """
        Initializes problem object. Calls base constructor. Sets
        nn.CrossEntropyLoss() as default loss function.

        :param params: Dictionary of parameters (read from configuration file).

        """
        super(VideoToClassProblem, self).__init__(params)

        # Set default loss function - cross entropy.
        self.loss_function = nn.CrossEntropyLoss()

    def calculate_accuracy(self, data_tuple, logits, aux_tuple):
        """ Calculates accuracy equal to mean number of correct predictions in a given batch.
        WARNING: Applies mask (from aux_tuple) to logits!

        :param logits: Logits being output of the model.
        :param data_tuple: Data tuple containing inputs and targets.
        :param aux_tuple: Auxiliary tuple containing mask.
        """
        # Mask logits (ONLY LOGITS!)
        masked_logits = logits[:, aux_tuple.mask, :][:, 0, :]

        # Get the index of the max log-probability.
        pred = masked_logits.max(1, keepdim=True)[1]
        correct = pred.eq(data_tuple.targets.view_as(pred)).sum().item()

        # Calculate the accuracy.
        batch_size = logits.size(0)
        accuracy = correct / batch_size

        return accuracy

    def evaluate_loss(self, data_tuple, logits, aux_tuple):
        """ Calculates loss.
        WARNING: Applies mask (from aux_tuple) to logits!

        :param logits: Logits being output of the model.
        :param data_tuple: Data tuple containing inputs and targets.
        :param aux_tuple: Auxiliary tuple containing mask.
        """
        # Mask logits (ONLY LOGITS!)
        masked_logits = logits[:, aux_tuple.mask, :][:, 0, :]

        # Unpack the data tuple.
        (_, targets) = data_tuple

        # Compute loss using the provided loss function.
        loss = self.loss_function(masked_logits, targets)

        return loss

    def add_statistics(self, stat_col):
        """
        Add accuracy statistic to collector.

        :param stat_col: Statistics collector.

        """
        stat_col.add_statistic('acc', '{:12.10f}')

    def collect_statistics(self, stat_col, data_tuple, logits, aux_tuple):
        """
        Collects accuracy.

        :param stat_col: Statistics collector.
        :param data_tuple: Data tuple containing inputs and targets.
        :param logits: Logits being output of the model.
        :param aux_tuple: auxiliary tuple (aux_tuple) is not used in this function.

        """
        stat_col['acc'] = self.calculate_accuracy(
            data_tuple, logits, aux_tuple)

    def show_sample(self, inputs, targets):
        import matplotlib.pyplot as plt

        # show data.
        plt.xlabel('num_columns')
        plt.ylabel('num_rows')
        plt.title('Target class:' + str(int(targets[0])))

        plt.imshow(inputs, interpolation='nearest', aspect='auto')
        # Plot!
        plt.show()
