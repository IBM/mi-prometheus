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
question_encoder.py: Implementation of the Question Encoder for the VWM Network
the reference paper.
"""
__author__ = "Vincent Albouy"
import torch
from torch.nn import Module
import torch.nn as nn

class QuestionEncoder(Module):
    """
    Implementation of the ``QuestionEncoder`` of the VWM network.
    """

    def __init__(self, vocabulary_size, dtype, words_embed_length, embedded_dim, dim):
        """
        Constructor for the ``QuestionEncoder``.
        
        :param vocabulary_size: size of dictionnary
        :type vocabulary_size: int
        
        :param dtype: dtype
        :type dtype: dtype
             
        :param words_embed_length: size of embbedings dim
        :type words_embed_length: int

        :param dim: global 'd' hidden dimension
        :type dim: int

        :param embedded_dim: dimension of the word embeddings.
        :type embedded_dim: int

        """

        # call base constructor
        super(QuestionEncoder, self).__init__()

        self.dim = dim

        self.dtype=dtype

        # create bidirectional LSTM layer
        self.lstm = torch.nn.LSTM(input_size=embedded_dim, hidden_size=self.dim,
                            num_layers=1, batch_first=True, bidirectional=True)

        # linear layer for projecting the word encodings from 2*dim to dim
        self.lstm_proj = torch.nn.Linear(2 * self.dim, self.dim)

        # Defines nn.Embedding for embedding of questions into float tensors.
        self.Embedding = nn.Embedding(vocabulary_size, words_embed_length, padding_idx=0)


    def forward(self, questions, questions_len):
        """
        Forward pass of the ``QuestionEncoder``.

        :param questions: tensor of the questions words, shape [batch_size x maxQuestionLength x embedded_dim].
        :type questions: torch.tensor

        :param questions_len: Unpadded questions length.
        :type questions_len: list

        :return question encodings: [batch_size x 2*dim] 
        :type: question encodings: torch.tensor
        
        :return: contextual_word_embedding: [batch_size x maxQuestionLength x dim] 
        :type: contextual_word_embedding: torch.tensor 
          
        """
        #get batch size
        batch_size = questions.shape[0]

        # Embeddings.
        embedded_questions = self.Embedding(questions)

        # LSTM layer: words & questions encodings
        lstm_out, (h, _) = self.lstm(embedded_questions)

        # get final words encodings using linear layer
        contextual_word_embedding = self.lstm_proj(lstm_out)

        # reshape last hidden states for questions encodings -> [batch_size x (2*dim)]
        question_encoding = h.permute(1, 0, 2).contiguous().view(batch_size, -1)

        return  contextual_word_embedding, question_encoding





