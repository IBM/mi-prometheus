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
masked_bce_with_logits_loss.py: contains masked binary cross entropy loss function.
"""
__author__ = "Ryan L. McAvoy"

import torch
from torch.nn import Module
from miprometheus.utils.app_state import AppState


class MaskedBCEWithLogitsLoss(Module):
    """
    Calculates the binary cross entropy for batches with different numbers of
    outputs for the samples.
    """

    def __init__(self, weight=None):
        """
        Constructor for the ``MaskedBCEWithLogitsLoss``.

        Defines the inner loss as ``BCEWithLogitsLoss``.

        :param weight: a manual rescaling weight given to each class. \
           If given, has to be a Tensor of size `C`

        :type weight: Tensor, optional

        """
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.loss_function = torch.nn.BCEWithLogitsLoss(reduce=False)
        # for pytorch 4.1
        # self.loss_function = torch.nn.BCELossWithLogits(reduction = "none")

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

        # if the loss has one extra dimenison then you need an extra unit dimension
        # to multiply element by element
        mask_float = mask.type(AppState().dtype)
        if len(mask.shape) < len(loss_per_element.shape):
            mask_float = mask_float.unsqueeze(-1)

        # Set the loss per element to zero for unneeded output
        masked_loss_per = mask_float * loss_per_element

        # obtain the number of non-zero elements in the mask.
        # nonzero() returns the indices so you have to divide by the number of dimensions
        # The mask lacks the last dimension of the targets so needs to be
        # scaled up
        size = mask.nonzero().numel() / len(mask.shape) * logits.shape[-1]

        loss = torch.sum(masked_loss_per) / size

        return loss

    def masked_accuracy(self, logits, targets, mask):
        """
        Calculates accuracy equal to mean number of correct predictions in a \
        given batch.

        .. warning::

            Applies ``mask`` to both ``logits`` and ``targets``.


        :param logits: Logits being output by the model. [batch, classes, sequence].
        :type logits: torch.tensor.

        :param targets: Targets [batch, sequence].
        :type targets: torch.LongTensor

        :param mask: Mask [batch, sequence].
        :type mask: torch.ByteTensor

        :return: accuracy value.

        """

        # calculate the accuracy per bit in the sequences
        acc_per = 1 - torch.abs(torch.round(torch.nn.functional.sigmoid(logits)) - targets)

        mask_float = mask.type(AppState().dtype)
        if len(mask.shape) < len(logits.shape):
            mask_float = mask_float.unsqueeze(-1)

        # The mask lacks the last dimension of the targets so needs to be
        # scaled up
        size = mask.nonzero().numel() / len(mask.shape) * logits.shape[-1]

        masked_acc_per = mask_float * acc_per

        accuracy = masked_acc_per.sum().item() / size

        return accuracy
