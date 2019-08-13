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
attention_module.py
"""
__author__ = "Vincent Albouy, T.S. Jayram"

import torch
from torch.nn import Module

from miprometheus.models.VWM_model.utils_VWM import linear


class AttentionModule(Module):
    """
    Implementation of the Attention_Module
    Uses a simple weighted dot product similarity function
    """

    def __init__(self, dim):
        """
        Constructor for the Attention Module

        :param dim: common dimension of query vector and keys
        :type dim: int
        """

        # call base constructor
        super(AttentionModule, self).__init__()

        # define the single layer perceptron used to create the attention weights.
        self.attn = torch.nn.Sequential(linear(dim, 1, bias=False),
                                        torch.nn.Softmax(dim=1))
        self.dim = dim

    def forward(self, query, keys, values=None):
        """
        Forward pass of the ``VWM model Attention_Module``.

        :param  query : query     [batch_size x dim]
        :param keys : Keys        [batch_size x N x dim]
        :param values : Values    [batch_size x N x dim_other]

        :return: c : content      [batch_size x dim_other]
        :return: ca : attention   [batch_size x N]

        """

        if values is None:
            values = keys

        assert query.size(-1) == self.dim, 'Dimension mismatch in query'
        assert keys.size(-1) == self.dim, 'Dimension mismatch in keys'
        assert values.size(-2) == keys.size(-2),\
            'Number of entities mismatch between keys and values'

        # compute element-wise weighted product between query and keys
        # and normalize them

        ca = self.attn(query[:, None, :] * keys)  # [batch_size x N x 1]

        # compute content to be retrieved
        c = (ca * values).sum(1)     # [batch_size x dim_other]

        ca = ca.squeeze(-1)     # [batch_size x N]

        return c, ca
