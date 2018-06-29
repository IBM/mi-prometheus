#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of a GRU based encoder for text2text problems (e.g. translation)"""
__author__ = "Vincent Marois "

import torch
from torch import nn


class EncoderRNN(nn.Module):
    """GRU Encoder for Encoder-Decoder"""

    def __init__(self, input_voc_size, hidden_size):
        """
        Initializes an Encoder network based on a Gated Recurrent Unit.
        :param input_voc_size: size of the vocabulary set to be embedded by the Embedding layer.
        :param hidden_size: length of embedding vectors.
        """
        # call base constructor.
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size

        # Embedding: creates a look-up table of the embedding of a vocabulary set
        # (size: input_voc_size -> input_language.n_words) on vectors of size hidden_size.
        # adds 1 dimension to the shape of the tensor
        # WARNING: input must be of type LongTensor
        self.embedding = nn.Embedding(num_embeddings=input_voc_size, embedding_dim=hidden_size)

        # Apply a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
        # NOTE: default number of recurrent layers is 1
        # 1st parameter: expected number of features in the input -> same as hidden_size because of embedding
        # 2nd parameter: expected number of features in hidden state -> hidden_size.
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, input, hidden):
        """
        Runs the Encoder.
        :param input: tensor of indices, of size [batch_size x 1] (word by word looping)
        :param hidden: initial hidden state for each element in the input batch.
        Should be of size [1 x batch_size x hidden_size]
        :return: For every input word, the encoder outputs a vector and a hidden state,
                and uses the hidden state for the next input word.
                - output should be of size [batch_size x seq_len x hidden_size]: tensor containing the output
                features h_t from the last layer of the RNN, for each t.
                - hidden should be of size [1 x batch_size x hidden_size]: tensor containing the hidden state for
                t = seq_length
        """
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)