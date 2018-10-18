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
stacked_attention_layer: Implements the Attention Layer as described in section 3.3 of the following paper:

@article{DBLP:journals/corr/YangHGDS15,
  author    = {Zichao Yang and
               Xiaodong He and
               Jianfeng Gao and
               Li Deng and
               Alexander J. Smola},
  title     = {Stacked Attention Networks for Image Question Answering},
  journal   = {CoRR},
  volume    = {abs/1511.02274},
  year      = {2015},
  url       = {http://arxiv.org/abs/1511.02274},
  archivePrefix = {arXiv},
  eprint    = {1511.02274},
  timestamp = {Mon, 13 Aug 2018 16:47:25 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/YangHGDS15},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

"""
__author__ = "Vincent Marois & Younes Bouhadjar"

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.app_state import AppState


class StackedAttentionLayer(nn.Module):
    """
    Stacks several layers of ``Attention`` to enable multi-step reasoning.

    """

    def __init__(self, question_image_encoding_size,
                 key_query_size, num_att_layers=2):
        """
        Constructor of the ``StackedAttentionLayers`` class.

        :param question_image_encoding_size: Size of the images & questions encoding.
        :type question_image_encoding_size: int

        :param key_query_size: Size of the Key & Query, considered the same for both in this implementation.
        :type key_query_size: int

        :param num_att_layers: Number of ``AttentionLayer`` to use.
        :type num_att_layers: int

        """
        # call base constructor
        super(StackedAttentionLayer, self).__init__()

        # to visualize attention
        self.visualize_attention = None

        self.san = nn.ModuleList([AttentionLayer(question_image_encoding_size, key_query_size)] * num_att_layers)

    def forward(self, encoded_image, encoded_question):
        """
        Apply stacked attention.

        :param encoded_image: output of the image encoding (CNN + FC layer), should be of shape \
        [batch_size, width * height, num_channels_encoded_image]
        :type encoded_image: torch.tensor

        :param encoded_question: Last hidden layer of the LSTM, of shape [batch_size, question_encoding_size]
        :type encoded_question: torch.tensor

        :return: u: attention [batch_size, num_channels_encoded_image]

        """

        for att_layer in self.san:
            u, attention_prob = att_layer(encoded_image, encoded_question)

            if AppState().visualize:
                if self.visualize_attention is None:
                    self.visualize_attention = attention_prob

                # Concatenate output
                else:
                    self.visualize_attention = torch.cat([self.visualize_attention, attention_prob], dim=-1)

        return u


class AttentionLayer(nn.Module):
    """
    Implements one layer of the Stacked Attention mechanism.

    Reference: Section 3.3 of the paper cited above.

    """

    def __init__(self, question_image_encoding_size, key_query_size=512):
        """
        Constructor of the ``AttentionLayer`` class.

        :param question_image_encoding_size: Size of the images & questions encoding.
        :type question_image_encoding_size: int

        :param key_query_size: Size of the Key & Query, considered the same for both in this implementation.
        :type key_query_size: int

        """
        # call base constructor
        super(AttentionLayer, self).__init__()

        # fully connected layer to construct the key
        self.ff_image = nn.Linear(question_image_encoding_size, key_query_size)

        # fully connected layer to construct the query
        self.ff_ques = nn.Linear(question_image_encoding_size, key_query_size)

        # fully connected layer to construct the attention from the query and key
        self.ff_attention = nn.Linear(key_query_size, 1)

    def forward(self, encoded_image, encoded_question):
        """
        Applies one layer of stacked attention over the image & question.

        :param encoded_image: output of the image encoding (CNN + FC layer), should be of shape \
        [batch_size, width * height, num_channels_encoded_image]
        :type encoded_image: torch.tensor

        :param encoded_question: Last hidden layer of the LSTM, of shape [batch_size, question_encoding_size]
        :type encoded_question: torch.tensor

        :returns:
            - "Refined query vector" (weighted sum of the image vectors, combine with the question vector), \
            of shape [batch_size, num_channels_encoded_image]
            - Attention weights, todo: shape?

        """

        # Get the key
        key = self.ff_image(encoded_image)

        # Get the query, unsqueeze to be able to add the query to all channels
        query = self.ff_ques(encoded_question).unsqueeze(dim=1)
        weighted_key_query = F.tanh(key + query)

        # Get attention over the different layers
        weighted_key_query = self.ff_attention(weighted_key_query)
        attention_prob = F.softmax(weighted_key_query, dim=-2)

        vi_attended = (attention_prob * encoded_image).sum(dim=1)
        u = vi_attended + encoded_question

        return u, attention_prob
