#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2018 Kim Seonghyeon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ------------------------------------------------------------------------------
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

        :param dim: global 'd' dimension.
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

    def forward(self,attention, question_encodings, mem):
        """
        Forward pass of the ``OutputUnit``.

        :param mem_state: final memory state, shape [batch_size x dim]
        :type mem_state: torch.tensor

        :param question_encodings: questions encodings, shape [batch_size x (2*dim)]
        :type question_encodings: torch.tensor

        :return: probability distribution over the classes, [batch_size x nb_classes]

        """
        # cat memory state & questions encodings


        concat = torch.cat([question_encodings, mem], dim=1)
        #print(concat.size())

        # get logits
        logits = self.classifier(concat)

        return logits
