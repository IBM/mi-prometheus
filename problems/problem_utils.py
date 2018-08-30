#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""problem_utils.py: contains utils for problem classes such as loss functions"""
__author__      = "Ryan L. McAvoy"
import torch.nn as nn
import torch.nn.functional as F
import torch
from misc.app_state import AppState

class MaskedCrossEntropyLoss(nn.Module):
    #Calculates the cross entropy for batches with different numbers of outputs for the samples

    def __init__(self, weight=None, ignore_index=-100, use_mask = True):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.loss_function = nn.CrossEntropyLoss(reduce = False)
        self.loss_function2 = nn.CrossEntropyLoss()
        self.use_mask = use_mask
        #self.loss_function2 = nn.CrossEntropyLoss()
        #for pytorch 4.1
        #self.loss_function = nn.CrossEntropyLoss(reduction = "none")
    def forward(self, logits, targets, mask):
        loss_per_element = self.loss_function(logits, targets)

        #Have to convert the mask to floats to multiply by the loss
        mask_float = mask.type(AppState().dtype)
        if len(mask.shape) < len(loss_per_element):
            mask_float = mask_float.unsqueeze(-1)

        masked_loss_per = mask_float*loss_per_element
        #obtain the number of non-zero elements in the mask.
        # nonzero() returns the indices so you have to divide by the number of dimensions
        size= mask.nonzero().numel()/len(mask.shape)
	#size= masked_loss_per.nonzero().numel()/len(masked_loss_per.shape) 

        loss = torch.sum(masked_loss_per)/size

        masque = mask[0].type(torch.cuda.ByteTensor)
        a = logits[:,:, masque]
        b= targets[:, masque]
        loss2 = self.loss_function2(a,b)
        print(loss.item(), loss2.item(), loss.item()-loss2.item())
        return loss

    def masked_accuracy(self, logits, targets, mask):
        """ Calculates accuracy equal to mean number of correct predictions in a given batch.
        WARNING: Applies mask (from aux_tuple) to logits!

        :param logits: Logits being output of the model.
        :param targets: Data tuple containing inputs and targets.
        :param mask: Auxiliary tuple containing mask.
        """
        # Mask logits (ONLY LOGITS!)
        #masked_logits = logits[:, aux_tuple.mask, :][:, 0, :]

        # Get the index of the max log-probability.
        pred = logits.max(1, keepdim=True)[1]
        #print(pred.shape)
        #print(targets.shape)
        correct_per = pred.eq(targets.view_as(pred))

        if len(mask.shape) < len(targets.shape):
            mask = mask.unsqueeze(-1)

        masked_correct_per = correct_per*mask

        #the mask has the same number of elements as the target in this case
        size= mask.nonzero().numel()/len(mask.shape)

        accuracy = masked_correct_per.sum().item()/size

        return accuracy

class MaskedBCEWithLogitsLoss(nn.Module):
    #Calculates the binary cross entropy for batches with different numbers of outputs for the samples

    def __init__(self, weight=None):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.loss_function = nn.BCEWithLogitsLoss(reduce = False)
        self.loss_function2 = nn.BCEWithLogitsLoss()
        #for pytorch 4.1
        #self.loss_function = nn.BCELossWithLogits(reduction = "none")
    def forward(self, logits, targets, mask):
        loss_per_element = self.loss_function(logits, targets)
 
        #The mask lacks the last dimension of the targets so needs to be scaled up
        size = mask.nonzero().numel()/len(mask.shape)*logits.shape[-1]

        mask_float = mask.type(AppState().dtype)
        if len(mask.shape) < len(loss_per_element.shape):
            mask_float = mask_float.unsqueeze(-1)

        masked_loss_per = mask_float*loss_per_element
        #obtain the number of non-zero elements in the mask. 
        #nonzero() returns the indices so you have to divide by the number of dimensions
        #The mask lacks the last dimension of the targets so needs to be scaled up
        size = mask.nonzero().numel()/len(mask.shape)*logits.shape[-1]

        loss = torch.sum(masked_loss_per)/size

        masque = mask[0].type(torch.cuda.ByteTensor)
        a = logits[:, masque,:]
        b= targets[:, masque]
        loss2 = self.loss_function2(a,b)
        print(loss.item(), loss2.item(), loss.item()-loss2.item())

        return loss

    def masked_accuracy(self, logits, targets, mask):
        """ Calculate accuracy equal to mean difference between outputs and targets.
        WARNING: Applies mask (from aux_tuple) to both logits and targets!

        :param logits: Logits being output of the model.
        :param targets: Data tuple containing inputs and targets.
        :param mask: Auxiliary tuple containing mask.
        """

        acc_per =  1 - torch.abs(torch.round(F.sigmoid(logits)) - targets)

        mask_float = mask.type(AppState().dtype)
        if len(mask.shape) < len(logits.shape):
            mask_float = mask_float.unsqueeze(-1)

        #The mask lacks the last dimension of the targets so needs to be scaled up
        size = mask.nonzero().numel()/len(mask.shape)*logits.shape[-1]

        masked_acc_per = mask_float*acc_per
        
        accuracy = masked_acc_per.sum().item()/size

        return accuracy
