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
masked_cross_entropy_loss.py: contains masked cross entropy loss function.
"""
__author__ = "Ryan L. McAvoy"

import torch
from torch.nn import Module
from miprometheus.utils.app_state import AppState


class MaskedCrossEntropyLoss(Module):
    """
    Calculates the cross entropy for batches with different numbers of outputs \
    per samples.
    """

    def __init__(self, weight=None, ignore_index=-100):
        """
        Constructor for the ``MaskedCrossEntropyLoss``.

        Defines the inner loss as ``CrossEntropyLoss``.

        :param weight: a manual rescaling weight given to each class. \
           If given, has to be a Tensor of size `C`
        :type weight: Tensor, optional

        :param ignore_index: Specifies a target value that is ignored \
            and does not contribute to the input gradient.
        :type ignore_index: int, optional

        """
        super(MaskedCrossEntropyLoss, self).__init__()
        self.loss_function = torch.nn.CrossEntropyLoss(reduce=False)
        # for pytorch 4.1
        # self.loss_function = nn.CrossEntropyLoss(reduction = "none")

    def forward(self, logits, targets, mask):
        """
        Calculates loss accounting for different numbers of output per sample.

        :param logits: Logits being output by the model. [batch, classes, sequence].
        :type logits: torch.tensor.

        :param targets: Targets [batch, sequence].
        :type targets: torch.LongTensor

        :param mask: Mask [batch, sequence].
        :type mask: torch.ByteTensor

        :return: loss value.

        """

        # Calculate the loss per element in the sequence
        loss_per_element = self.loss_function(logits, targets)

        # Have to convert the mask to floats to multiply by the loss
        mask_float = mask.type(AppState().dtype)

        # if the loss has one extra dimenison then you need an extra unit dimension
        # to multiply element by element
        if len(mask.shape) < len(loss_per_element.shape):
            mask_float = mask_float.unsqueeze(-1)

        # Set the loss per element to zero for unneeded output
        masked_loss_per = mask_float * loss_per_element

        # obtain the number of non-zero elements in the mask.
        # nonzero() returns the indices so you have to divide by the number of
        # dimensions
        size = mask.nonzero().numel() / len(mask.shape)

        # add up the loss scaling by only the needed outputs
        loss = torch.sum(masked_loss_per) / size
        return loss

    def masked_accuracy(self, logits, targets, mask):
        """
        Calculates accuracy equal to mean number of correct predictions in a \
        given batch.

        :param logits: Logits being output by the model. [batch, classes, sequence].
        :type logits: torch.tensor.

        :param targets: Targets [batch, sequence].
        :type targets: torch.LongTensor

        :param mask: Mask [batch, sequence].
        :type mask: torch.ByteTensor

        :return: accuracy value.

        """
        # Get the index of the max log-probability.
        # Max returns a tuple and we only need the second element
        pred = logits.max(1)[1]
        correct_per = pred.eq(targets.view_as(pred))

        if len(mask.shape) < len(correct_per.shape):
            mask = mask.unsqueeze(-1)

        # set the unneeded outputs to zero
        masked_correct_per = correct_per * mask

        # scale by only the number of needed outputs
        # the mask has the same number of elements as the target in this case
        size = mask.nonzero().numel() / len(mask.shape)

        accuracy = masked_correct_per.sum().item() / size

        return accuracy

