#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""problem_utils.py: contains utils for problem classes such as loss functions"""
__author__      = "Ryan L. McAvoy"
import torch.nn as nn
import torch.nn.functional as F
import torch
from misc.app_state import AppState

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, use_mask = True):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.loss_function = nn.CrossEntropyLoss(reduce = False)
        self.loss_function2 = nn.CrossEntropyLoss()
        self.use_mask = use_mask
        #self.loss_function2 = nn.CrossEntropyLoss()
        #for pytorch 4.1
        #self.loss_function = nn.CrossEntropyLoss(reduction = "none")
    def forward(self, predictions, targets, mask):
        loss_per_element = self.loss_function(predictions, targets)
        
        masque = mask.type(AppState().dtype)
        if len(mask.shape) < len(loss_per_element):
            masque = masque.unsqueeze(-1)

        #if len(mask.shape) < len(predictions.shape):
        #    masque = mask.type(AppState().dtype).unsqueeze(-1)
        
        masked_loss_per = masque*loss_per_element
        #obtain the number of non-zero elements in the mask.
        # nonzero() returns the indices so you have to divide by the number of dimensions
        size= masque.nonzero().numel()/len(masque.shape)
	#size= masked_loss_per.nonzero().numel()/len(masked_loss_per.shape) 

        loss = torch.sum(masked_loss_per)/size

        masque = mask[0].type(torch.cuda.ByteTensor)
        a = predictions[:,:, masque]
        b= targets[:, masque]
        loss2 = self.loss_function2(a,b)
        print(loss.item(), loss2.item(), loss.item()-loss2.item())
        return loss

    def masked_accuracy(self, logits, targets, mask):
        """ Calculates accuracy equal to mean number of correct predictions in a given batch.
        WARNING: Applies mask (from aux_tuple) to logits!

        :param logits: Logits being output of the model.
        :param data_tuple: Data tuple containing inputs and targets.
        :param aux_tuple: Auxiliary tuple containing mask.
        """
        # Mask logits (ONLY LOGITS!)
        #masked_logits = logits[:, aux_tuple.mask, :][:, 0, :]

        # Get the index of the max log-probability.
        pred = logits.max(1, keepdim=True)[1]
        correct_per = pred.eq(data_tuple.targets.view_as(pred))

        masque = mask.type(AppState().dtype)
        if len(mask.shape) < len(correct_per.shape):
            masque = masque.unsqueeze(-1)

        masked_correct_per = correct_per*masque
 
        #the mask has the same number of elements as the target in this case
        size= masque.nonzero().numel()/len(masque.shape)

        accuracy = masked_correct_per.sum().item()/size

        return accuracy

class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.loss_function = nn.BCEWithLogitsLoss(reduce = False)
        self.loss_function2 = nn.BCEWithLogitsLoss()
        #for pytorch 4.1
        #self.loss_function = nn.BCELossWithLogits(reduction = "none")
    def forward(self, predictions, targets, mask):
        loss_per_element = self.loss_function(predictions, targets)
 
        #The mask lacks the last dimension of the targets so needs to be scaled up
        size = masque.nonzero().numel()/len(masque.shape)*logits.shape[-1]

        masque = mask.type(AppState().dtype)
        if len(mask.shape) < len(loss_per_element.shape):
            masque = masque.unsqueeze(-1)

        masked_loss_per = masque*loss_per_element
        #obtain the number of non-zero elements in the mask. 
        #nonzero() returns the indices so you have to divide by the number of dimensions
        #The mask lacks the last dimension of the targets so needs to be scaled up
        size = masque.nonzero().numel()/len(masque.shape)*logits.shape[-1]
        #size= masked_loss_per.nonzero().numel()/len(masked_loss_per.shape)

        loss = torch.sum(masked_loss_per)/size

        masque = mask[0].type(torch.cuda.ByteTensor)
        a = predictions[:, masque,:]
        b= targets[:, masque]
        loss2 = self.loss_function2(a,b)
        print(loss.item(), loss2.item(), loss.item()-loss2.item())

        return loss

    def masked_accuracy(self, logits, targets, mask):
        """ Calculate accuracy equal to mean difference between outputs and targets.
        WARNING: Applies mask (from aux_tuple) to both logits and targets!

        :param logits: Logits being output of the model.
        :param data_tuple: Data tuple containing inputs and targets.
        :param aux_tuple: Auxiliary tuple containing mask.
        """

        acc_per =  1 - torch.abs(torch.round(F.sigmoid(logits)) - targets)

        masque = mask.type(AppState().dtype)
        if len(mask.shape) < len(logits.shape):
            masque = masque.unsqueeze(-1)

        #The mask lacks the last dimension of the targets so needs to be scaled up
        size = masque.nonzero().numel()/len(masque.shape)*logits.shape[-1]

        masked_acc_per = masque*acc_per

        #print(acc_per.nonzero().numel()/len(acc_per.shape))

        #orig_size_zeros = acc_per.numel() -  acc_per.nonzero().numel()/len(acc_per.shape)
        #print(orig_size_zeros, acc_per.numel())

        accuracy = masked_acc_per.sum().item()/size

        return accuracy
