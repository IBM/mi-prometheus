#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""problem_utils.py: contains utils for problem classes such as loss functions"""
__author__      = "Ryan L. McAvoy"
import torch.nn as nn
import torch.nn.functional as F
import torch
from misc.app_state import AppState

class MaskedCrossEntropyLoss(nn.Module):
    """Calculates the cross entropy for batches with different numbers of outputs per
    """

    def __init__(self, weight=None, ignore_index=-100):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.loss_function = nn.CrossEntropyLoss(reduce = False)
        #for pytorch 4.1
        #self.loss_function = nn.CrossEntropyLoss(reduction = "none")
    def forward(self, logits, targets, mask):
        """ Calculates loss accounting for different numbers of output per sample

        :param logits: Logits being output by the model. [batch, classes, sequence]
        :param targets: LongTensor targets [batch, sequence]
        :param mask: ByteTensor mask [batch, sequence]
        """

        # Calculate the loss per element in the sequence
        loss_per_element = self.loss_function(logits, targets)

        #Have to convert the mask to floats to multiply by the loss
        mask_float = mask.type(AppState().dtype)

        #if the loss has one extra dimenison then you need an extra unit dimension 
        #to multiply element by element 
        if len(mask.shape) < len(loss_per_element.shape):
            mask_float = mask_float.unsqueeze(-1)

        # Set the loss per element to zero for unneeded output
        masked_loss_per = mask_float*loss_per_element

        #obtain the number of non-zero elements in the mask.
        # nonzero() returns the indices so you have to divide by the number of dimensions
        size= mask.nonzero().numel()/len(mask.shape)

        # add up the loss scaling by only the needed outputs
        loss = torch.sum(masked_loss_per)/size
        return loss

    def masked_accuracy(self, logits, targets, mask):
        """ Calculates accuracy equal to mean number of correct predictions in a given batch.

        :param logits: Logits being output by the model. [batch, classes, sequence]
        :param targets: LongTensor targets [batch, sequence]
        :param mask: ByteTensor mask [batch, sequence]

        """
        # Get the index of the max log-probability. 
        #Max returns a tuple and we only need the second element
        pred = logits.max(1)[1]
        correct_per = pred.eq(targets.view_as(pred))

        if len(mask.shape) < len(correct_per.shape):
            mask = mask.unsqueeze(-1)

        # set the unneeded outputs to zero
        masked_correct_per = correct_per*mask

        #scale by only the number of needed outputs
        #the mask has the same number of elements as the target in this case
        size= mask.nonzero().numel()/len(mask.shape)

        accuracy = masked_correct_per.sum().item()/size

        return accuracy

class MaskedBCEWithLogitsLoss(nn.Module):
    """Calculates the binary cross entropy for batches with different numbers of outputs for the samples
    """

    def __init__(self, weight=None):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.loss_function = nn.BCEWithLogitsLoss(reduce = False)
        #for pytorch 4.1
        #self.loss_function = nn.BCELossWithLogits(reduction = "none")
    def forward(self, logits, targets, mask):
        """ Calculates loss accounting for different numbers of output per sample

        :param logits: Logits being output by the model. [batch, sequence, element_size]
        :param targets: LongTensor targets [batch, sequence, element_size]
        :param mask: ByteTensor mask [batch, sequence]
        """

        # Calculate the loss per element in the sequence
        loss_per_element = self.loss_function(logits, targets)

        #obtain the number of non-zero elements in the mask.
        #The mask lacks the last dimension of the targets so needs to be scaled up
        size = mask.nonzero().numel()/len(mask.shape)*logits.shape[-1]

        #if the loss has one extra dimenison then you need an extra unit dimension 
        #to multiply element by element 
        mask_float = mask.type(AppState().dtype)
        if len(mask.shape) < len(loss_per_element.shape):
            mask_float = mask_float.unsqueeze(-1)

        # Set the loss per element to zero for unneeded output
        masked_loss_per = mask_float*loss_per_element
        #obtain the number of non-zero elements in the mask. 
        #nonzero() returns the indices so you have to divide by the number of dimensions
        #The mask lacks the last dimension of the targets so needs to be scaled up
        size = mask.nonzero().numel()/len(mask.shape)*logits.shape[-1]

        loss = torch.sum(masked_loss_per)/size
 
        return loss

    def masked_accuracy(self, logits, targets, mask):
        """ Calculate accuracy equal to mean difference between outputs and targets.
        WARNING: Applies mask (from aux_tuple) to both logits and targets!

        :param logits: Logits being output by the model. [batch, sequence, element_size]
        :param targets: LongTensor targets [batch, sequence, element_size]
        :param mask: ByteTensor mask [batch, sequence]
        """

        #calculate the accuracy per bit in the sequences
        acc_per =  1 - torch.abs(torch.round(F.sigmoid(logits)) - targets)

        mask_float = mask.type(AppState().dtype)
        if len(mask.shape) < len(logits.shape):
            mask_float = mask_float.unsqueeze(-1)

        #The mask lacks the last dimension of the targets so needs to be scaled up
        size = mask.nonzero().numel()/len(mask.shape)*logits.shape[-1]

        masked_acc_per = mask_float*acc_per

        accuracy = masked_acc_per.sum().item()/size

        return accuracy
