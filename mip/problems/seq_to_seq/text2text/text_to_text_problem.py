#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# The MIT License (MIT)
#
# Copyright (c) 2017 Sean Robertson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
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
text_to_text_problem.py: abstract base class for text to text sequential problems, e.g. machine translation.

"""

__author__ = "Vincent Marois"

import unicodedata
import re
import torch
import torch.nn as nn
from mip.problems.seq_to_seq.seq_to_seq_problem import SeqToSeqProblem

# global tokens
PAD_token = 0
SOS_token = 1
EOS_token = 2


class TextToTextProblem(SeqToSeqProblem):
    """
    Base class for text to text sequential problems.

    Provides some basic features useful in all problems of such
    type.

    """

    def __init__(self, params):
        """
        Initializes problem object. Calls base ``SeqToSeqProblem`` constructor.

        Sets ``nn.NLLLoss()`` as default loss function.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).

        """
        super(TextToTextProblem, self).__init__(params)

        # set default loss function - negative log likelihood and ignore
        # padding elements.
        self.loss_function = nn.NLLLoss(size_average=True, ignore_index=0)

        # set default data_definitions dict
        self.data_definitions = {'inputs': {'size': [-1, -1, -1], 'type': [torch.Tensor]},
                                 'inputs_length': {'size': [-1, 1], 'type': [list, int]},
                                 'inputs_text': {'size': [-1, 1], 'type': [list, str]},
                                 'targets': {'size': [-1, -1, -1], 'type': [torch.Tensor]},
                                 'targets_length': {'size': [-1, 1], 'type': [list, int]},
                                 'outputs_text': {'size': [-1, 1], 'type': [list, str]},
                                 }

        # default values likely to be useful to a model.
        # setting the fields for the vocabulary sets sizes to None for now.
        # TODO: other fields to consider?
        self.default_values = {'input_voc_size': None,
                               'output_voc_size': None,
                               'embedding_dim': None}

        self.input_lang = None
        self.output_lang = None

    def compute_BLEU_score(self, data_dict, logits):
        """
        Compute the BLEU score in order to evaluate the translation quality
        (equivalent of accuracy).

        .. note::

            Reference paper: http://www.aclweb.org/anthology/P02-1040.pdf

            Implementation inspired from https://machinelearningmastery.com/calculate-bleu-score-for-text-python/


            To handle all samples within a batch, we accumulate the individual BLEU score for each pair\
             of sentences and average over the batch size.


        :param data_dict: DataDict({'inputs', 'inputs_length', 'inputs_text', 'targets', 'targets_length', 'outputs_text'}).

        :param logits: Predictions of the model.

        :return: Average BLEU Score for the batch ( 0 < BLEU < 1).

        """
        # get most probable words indexes for the batch
        _, top_indexes = logits.topk(k=1, dim=-1)
        logits = top_indexes.squeeze()
        batch_size = logits.shape[0]

        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        # retrieve target sentences from TextAuxTuple
        targets_text = []
        for sentence in data_dict['targets_text']:
            targets_text.append(sentence.split())

        # retrieve text sentences from the logits (which should be tensors of
        # indexes)
        logits_text = []
        for logit in logits:
            logits_text.append(
                [self.output_lang.index2word[index.item()] for index in logit])

        bleu_score = 0
        for i in range(batch_size):
            # compute bleu score and use a smoothing function
            bleu_score += sentence_bleu([targets_text[i]],
                                        logits_text[i],
                                        smoothing_function=SmoothingFunction().method1)

        return round(bleu_score / batch_size, 4)

    def evaluate_loss(self, data_dict, logits):
        """
        Computes loss.

        By default, the loss function is the Negative Log Likelihood function.

        The input given through a forward call is expected to contain log-probabilities (LogSoftmax) of each class.

        The input has to be a Tensor of size either (batch_size, C) or (batch_size, C, d1, d2,...,dK) with K ≥ 2 for the
        K-dimensional case.

        The target that this loss expects is a class index (0 to C-1, where C = number of classes).

        :param data_dict: DataDict({'inputs', 'inputs_length', 'inputs_text', 'targets', 'targets_length', 'outputs_text'}).

        :param logits: Predictions of the model.

        :return: loss
        """
        loss = self.loss_function(logits.transpose(1, 2), data_dict['targets'])

        return loss

    def add_statistics(self, stat_col):
        """
        Add BLEU score to a ``StatisticsCollector``.

        :param stat_col: Statistics collector.
        :type stat_col: ``StatisticsCollector``

        """
        stat_col.add_statistic('bleu_score', '{:4.5f}')

    def collect_statistics(self, stat_col, data_dict, logits):
        """
        Collects BLEU score.

        :param stat_col: ``StatisticsCollector``

        :param data_dict: DataDict({'inputs', 'inputs_length', 'inputs_text', 'targets', 'targets_length', 'outputs_text'}).

        :param logits: Predictions of the model.

        """

        stat_col['bleu_score'] = self.compute_BLEU_score(data_dict, logits)

    def show_sample(self, data_dict, sample=0):
        """
        Shows the sample (both input and target sequences) using matplotlib.
        Elementary visualization.

        :param data_dict: DataDict({'inputs', 'inputs_length', 'inputs_text', 'targets', 'targets_length', 'outputs_text'}).

        :param sample: Number of sample in a batch (default: 0)

        .. note::

            TODO


        """
        pass

    # ----------------------
    # The following are helper functions for data pre-processing in the case
    # of a translation task

    def unicode_to_ascii(self, s):
        """
        Turn a Unicode string to plain ASCII.

        See: http://stackoverflow.com/a/518232/2809427.

        :param s: Unicode string.

        :return: plain ASCII string.

        """
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalize_string(self, s):
        """
        Lowercase, trim, and remove non-letter characters in string s.

        :param s: string.

        :return: normalized string.

        """
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def indexes_from_sentence(self, lang, sentence):
        """
        Construct a list of indexes using a 'vocabulary index' from a specified
        Lang class instance for the specified sentence (see ``Lang`` class below).

        :param lang: instance of the ``Lang`` class, having a ``word2index`` dict.
        :type lang: Lang

        :param sentence: string to convert word for word to indexes, e.g. "The black cat is eating."
        :type sentence: str

        :return: list of indexes.

        """
        seq = [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]

        return seq

    def tensor_from_sentence(self, lang, sentence):
        """
        Uses ``indexes_from_sentence()`` to create a tensor of indexes with the
        EOS token.

        :param lang: instance of the ``Lang`` class, having a ``word2index`` dict.
        :type lang: Lang

        :param sentence: string to convert word for word to indexes, e.g. "The black cat is eating."
        :type sentence: str

        :return: tensor of indexes, terminated by the EOS token.

        """
        indexes = self.indexes_from_sentence(lang, sentence)

        return torch.tensor(indexes).type(self.app_state.LongTensor)

    def tensors_from_pair(self, pair, input_lang, output_lang):
        """
        Creates a tuple of tensors of indexes from a pair of sentences.

        :param pair: input & output languages sentences
        :type pair: tuple

        :param input_lang: instance of the ``Lang`` class, having a ``word2index`` dict, representing the input language.
        :type lang: Lang

        :param output_lang: instance of the ``Lang`` class, having a ``word2index`` dict, representing the output language.
        :type lang: Lang

        :return: tuple of tensors of indexes.

        """
        input_tensor = self.tensor_from_sentence(input_lang, pair[0])
        target_tensor = self.tensor_from_sentence(output_lang, pair[1])

        return [input_tensor, target_tensor]

    def tensors_from_pairs(self, pairs, input_lang, output_lang):
        """
        Returns a list of tuples of tensors of indexes from a list of pairs of
        sentences. Uses ``tensors_from_pair()``.

        :param pairs: sentences pairs
        :type pairs: list

        :param input_lang: instance of the class Lang, having a word2index dict, representing the input language.
        :type lang: Lang

        :param output_lang: instance of the class Lang, having a word2index dict, representing the output language.
        :type lang: Lang

        :return: list of tensors of indexes.

        """
        return [self.tensors_from_pair(pair, input_lang, output_lang) for pair in pairs]


class Lang(object):
    """
    Simple helper class allowing to represent a language in a translation task.
    It will contain for instance a vocabulary index (``word2index`` dict) & keep
    track of the number of words in the language.

    This class is useful as each word in a language will be represented as a one-hot vector: a giant vector of zeros
    except for a single one (at the index of the word). The dimension of this vector is potentially very high, hence it
    is generally useful to trim the data to only use a few thousand words per language.

    The inputs and targets of the associated sequence to sequence networks will be sequences of indexes, each item
    representing a word. The attributes of this class (``word2index``, ``index2word``, ``word2count``) are useful to\
     keep track of this.

    """

    def __init__(self, name):
        """
        Constructor.

        :param name: string to name the language (e.g. french, english)

        """
        self.name = name

        self.word2index = {"PAD": 0, "SOS": 1, "EOS": 2}  # dict 'word': index
        # keep track of the occurrence of each word in the language. Can be
        # used to replace rare words.
        self.word2count = {}

        # dict 'index': 'word', initializes with PAD, EOS, SOS tokens
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}

        # Number of words in the language. Start by counting PAD, EOS, SOS
        # tokens.
        self.n_words = 3

    def add_sentence(self, sentence):
        """
        Process a sentence using ``add_word()``.

        :param sentence: sentence to be added to the language.
        :type sentence: str

        """
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        """
        Add a word to the vocabulary set: update word2index, word2count,
        index2words & n_words.

        :param word: word to be added.
        :type word: str

        """

        if word not in self.word2index:  # if the current word has not been seen before
            # create a new entry in word2index
            self.word2index[word] = self.n_words
            self.word2count[word] = 1  # count first occurrence of this word
            # create a new entry in index2word
            self.index2word[self.n_words] = word
            self.n_words += 1  # increment total number of words in the language

        else:  # this word has been seen before, simply update its occurrence
            self.word2count[word] += 1
