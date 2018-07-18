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

"""attention.py: implement different attention model: for now just the stacked attention exists https://arxiv.org/abs/1511.02274
 for future co-attention needs to be implemented https://arxiv.org/abs/1707.04968 """
__author__ = "Younes Bouhadjar"

import torch.nn as nn
import torch.nn.functional as F


class StackedAttention(nn.Module):
    def __init__(self, question_image_encoding_size, key_query_size, num_att_layers=2):
        """
        Implements a stacked attention layers, this attention is designed in this paper: https://arxiv.org/abs/1511.02274

        :param question_image_encoding_size: question encoding size.
        :param key_query_size: key and query size, are the same in this implementation
        :param num_att_layers: num of stacked attention layers
        """

        super(StackedAttention, self).__init__()

        self.san = nn.ModuleList(
            [Attention(question_image_encoding_size, key_query_size)] * num_att_layers)

    def forward(self, encoded_image, encoded_question):
        """
        Apply stacked attention

        :param encoded_image: output of the image encoding (CNN + FC layer), [batch_size, new_width * new_height, num_channels_encoded_image]
        :param encoded_question: last hidden layer of the LSTM, [batch_size, question_encoding_size]
        :return u: attention [batch_size, num_channels_encoded_image]
        """

        for att_layer in self.san:
            u = att_layer(encoded_image, encoded_question)

        return u


class Attention(nn.Module):
    def __init__(self, question_image_encoding_size, key_query_size=512):
        """
        Implements one layer of the stacked attention designed: https://arxiv.org/abs/1511.02274

        :param question_image_encoding_size: question encoding size.
        :param key_query_size: key and query size, are the same in this implementation
        """

        super(Attention, self).__init__()
        # fully connected layer to construct the key
        self.ff_image = nn.Linear(question_image_encoding_size, key_query_size)
        # fully connected layer to construct the query
        self.ff_ques = nn.Linear(question_image_encoding_size, key_query_size)
        # fully connected layer to construct the attention from the query and key
        self.ff_attention = nn.Linear(key_query_size, 1)

    def forward(self, encoded_image, encoded_question):
        """
        Apply a single attention layer

        :param encoded_image: output of the image encoding (CNN + FC layer), [batch_size, new_width * new_height, num_channels_encoded_image]
        :param encoded_question: last hidden layer of the LSTM, [batch_size, question_encoding_size]
        :return u: attention [batch_size, num_channels_encoded_image]
        """

        # Get the key
        key = self.ff_image(encoded_image)

        # Get the query, unsqueeze to be able to add the query to all channels
        query = self.ff_ques(encoded_question).unsqueeze(dim=1)
        weighted_key_query = F.tanh(key + query)

        # Get attention over the different layers
        weighted_key_query = self.ff_attention(weighted_key_query)
        attention_prob = F.softmax(weighted_key_query, dim=-2)

        # sum the weighted channels
        vi_attended = (attention_prob * encoded_image).sum(dim=1)
        u = vi_attended + encoded_question

        return u