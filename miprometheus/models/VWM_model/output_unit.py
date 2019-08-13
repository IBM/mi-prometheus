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

"""
output_unit.py: Implementation of the ``OutputUnit`` for the MAC network. Cf https://arxiv.org/abs/1803.03067 \
for the reference paper.
"""
__author__ = "Vincent Albouy, T.S. Jayram"

import torch
from torch.nn import Module

from miprometheus.models.VWM_model.utils_VWM import linear


class OutputUnit(Module):
    """
    Implementation of the ``OutputUnit`` of the MAC network.
    """

    def __init__(self, dim, nb_classes):
        """
        Constructor for the ``OutputUnit``.

        :param dim: dimension of feature vectors
        :type dim: int

        :param nb_classes: number of classes to consider (classification problem).
        :type nb_classes: int

        """

        # call base constructor
        super(OutputUnit, self).__init__()

        # define the 2-layers MLP & specify weights initialization
        self.classifier = torch.nn.Sequential(linear(384, 3*dim, bias=True),
                                              torch.nn.ELU(),
                                              linear(3*dim, nb_classes, bias=True))
        torch.nn.init.kaiming_uniform_(self.classifier[0].weight)

    def forward(self, question_encoding, summary_object):
        """
        Forward pass of the ``OutputUnit``.

        :param question_encoding: questions encodings      [batch_size x (2*dim)]
        :param summary_object: final summary object         [batch_size x dim]

        :return logits: scores over the classes             [batch_size x nb_classes]

        """

        concat = torch.cat([question_encoding, summary_object], dim=1)
        logits = self.classifier(concat)

        return logits
