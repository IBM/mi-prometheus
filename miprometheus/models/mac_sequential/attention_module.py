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
attention_module.py
"""
__author__ = "Vincent Albouy"

import torch
from torch.nn import Module
from miprometheus.models.mac_sequential.utils_mac import linear


class Attention_Module(Module):
    """
    Implementation of the Attention_Module for VWM model 
    """

    def __init__(self, dim):
        """
        Constructor for the VWM model Attention_Module

        :param dim: global 'd' hidden dimension
        :type dim: int

        """

        # call base constructor
        super(Attention_Module, self).__init__()


        # define the linear layer used to create the attention weights. Should
        # be one scalar weight per contextual word
        self.attn = linear(dim, 1, bias=True)

    def forward(self,q, K):
        """
        Forward pass of the ``VWM model Attention_Module``.

        :param  q : attention query 
        :type   tensor

        :param K : attention Keys
        :type  tensor

        :return: c : content , ca : attention
        :type  tensors 
        
        """

        # compute element-wise product between q & k
        # compute attention weights


        cai = self.attn(q[:,None,:] * K)  # [batch_size x maxLength x dim]

        # get attention distribution
        ca = torch.nn.functional.softmax(cai, dim=1)   # [batch_size x maxLength]

        # compute content
        c = (ca * K).sum(1) # [batch_size x dim]

        #return content and attention tensors
        return c, ca
