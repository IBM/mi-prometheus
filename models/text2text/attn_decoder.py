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

"""attn_decoder.py: Implementation of a GRU based attention decoder for text2text problems (e.g. translation)
    Inspiration taken from the corresponding Pytorch tutorial.
    See https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html """
__author__ = "Vincent Marois "

import torch
from torch import nn
import torch.nn.functional as F


class AttnDecoderRNN(nn.Module):
    """GRU Attention Decoder for Encoder-Decoder"""

    def __init__(self, hidden_size, output_voc_size, dropout_p=0.1,
                 max_length=10, encoder_bidirectional=True):
        """
        Initializes an Decoder network based on a Gated Recurrent Unit.
        :param hidden_size: length of embedding vectors.
        :param output_voc_size: size of the vocabulary set to be embedded by the Embedding layer.
        :param dropout_p: probability of an element to be zeroed for the Dropout layer.
        :param max_length: maximum sequence length.
        :param encoder_bidirectional: whether the associated encoder is bidirectional or not.
        """
        # call base constructor
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_voc_size = output_voc_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.encoder_bidirectional = encoder_bidirectional

        # Embedding: creates a look-up table of the embedding of a vocabulary set
        # (size: output_voc_size -> output_language.n_words) on vectors of size hidden_size.
        # adds 1 dimension to the shape of the tensor
        # WARNING: input must be of type LongTensor
        self.embedding = nn.Embedding(self.output_voc_size, self.hidden_size)

        # Apply a linear transformation to the incoming data: y=Ax+b
        # used to create the attention weights
        if self.encoder_bidirectional:
            self.attn = nn.Linear(self.hidden_size * 3, self.max_length)
        else:
            self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        # Apply a linear transformation to the incoming data: y=Ax+b
        # used to combine the embedded decoder inputs & the attented encoder
        # outputs
        if self.encoder_bidirectional:
            self.attn_combine = nn.Linear(
                self.hidden_size * 3, self.hidden_size)
        else:
            self.attn_combine = nn.Linear(
                self.hidden_size * 2, self.hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_p)

        # Apply a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
        # NOTE: default number of recurrent layers is 1
        # 1st parameter: expected number of features in the input -> same as hidden_size because of embedding
        # 2nd parameter: expected number of features in hidden state -> hidden_size.
        # batch_first=True -> input and output tensors are provided as (batch, seq, feature)
        # batch_first=True do not affect hidden states
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True)

        # Apply a linear transformation to the incoming data: y=Ax+b
        # basically project from the hidden space to the output vocabulary set
        self.out = nn.Linear(self.hidden_size, self.output_voc_size)

    def forward(self, input, hidden, encoder_outputs):
        """
        Runs the Attention Decoder.

        :param input: tensor of indices, of size [batch_size x 1] (word by word looping)

        :param hidden: initial hidden state for each element in the input batch. Should be of size [1 x batch_size x hidden_size].

        :param encoder_outputs: encoder outputs, of shape [batch_size x max_length x hidden_size]

        :return: output should be of size [batch_size x 1 x output_voc_size]: tensor containing the output features h_t from the last layer of the RNN, for each t.

        :return: hidden should be of size [1 x batch_size x hidden_size]: tensor containing the hidden state for t = seq_length

        """
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        # embedded: [batch_size x 1 x hidden_size]

        # concatenate embedded decoder inputs & hidden states
        batch_size = input.shape[0]
        if self.encoder_bidirectional:  # flatten out hidden states if the encoder was bidirectional
            hidden = hidden.view(1, batch_size, -1)
        embedded_hidden_concat = torch.cat(
            (embedded, hidden.transpose(0, 1)), dim=-1)
        # embedded_hidden_concat: [batch_size x 1 x (hidden_size * (1 +
        # encoder.n_dir)]

        # compute attention weights
        attn_weights = self.attn(embedded_hidden_concat)
        attn_weights = F.softmax(attn_weights, dim=-1)
        # attn_weights: [batch_size x 1 x max_length]

        # apply attention weights on the encoder outputs
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        # attn_applied: [batch_size x 1 x (hidden_size * (1 + encoder.n_dir)]

        # combine the embedded decoder inputs & attended encoder outputs
        embedded_attn_applied_concat = torch.cat(
            (embedded, attn_applied), dim=-1)
        gru_input = self.attn_combine(embedded_attn_applied_concat)
        gru_input = F.relu(gru_input)

        if self.encoder_bidirectional:  # select hidden state of forward layer of encoder only
            hidden = hidden[:, :, :self.hidden_size]
        gru_output, hidden = self.gru(gru_input, hidden.contiguous())

        if self.encoder_bidirectional:  # 'hack' if the encoder is bidirectional: hidden states need to be of shape
            # [(n_layers * n_directions) x batch_size x hidden_size]
            hidden = torch.cat((hidden, hidden), dim=0)
        output = self.out(gru_output)
        output = F.log_softmax(output, dim=-1)

        return output, hidden, attn_weights
