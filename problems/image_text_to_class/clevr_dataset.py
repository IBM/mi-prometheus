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
clevr_dataset.py: This file contains 1 class:

- ClevrDataset (CLEVR object class): This class creates a Dataset object to represent a CLEVR dataset.
  ClevrDataset builds the embedding for the questions.


"""
__author__ = "Vincent Albouy, Vincent Marois"

import h5py
import torch
import pickle

from torch.utils.data import Dataset

import os

from problems.utils.language import Language
from misc.app_state import AppState
app_state = AppState()

import logging
logger = logging.getLogger('CLEVR')


class CLEVRDataset(Dataset):
    """
    Inherits from the Dataset class to represent the CLEVR dataset. Will be used by the Clevr class to generate
    batches.

    ---------------
    class Dataset(object): An abstract class representing a Dataset.
    All other datasets should subclass it. All subclasses should override __len__, that provides the size of the dataset
    , and __getitem__, supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, set, clevr_dir, clevr_humans, embedding_type='random', random_embedding_dim=300):
        """
        Instantiate a ClevrDataset object:
            - Mainly check if the files containing the extracted features & tokenized questions already exist. If not,
            it generates them for the specified sub-set.
            - self.img contains then the extracted feature maps
            - self.data contains the tokenized questions, the associated image filenames, the answers & the question string

        The questions are then embedded based on the specified embedding. This embedding is random by default, but
        pretrained ones are possible.

        :param set: String to specify which dataset to use: 'train', 'val' or 'test'.
        :param clevr_dir: Directory path to the CLEVR_v1.0 dataset. Will also be used to store the generated files (.hdf5, .pkl)
        :param clevr_humans: Boolean to indicate whether to use the questions from CLEVR-Humans.

        :param embedding_type: string to indicate the pretrained embedding to use: either 'random' to use nn.Embedding
        or one of the following: "charngram.100d", "fasttext.en.300d", "fasttext.simple.300d", "glove.42B.300d",
        "glove.840B.300d", "glove.twitter.27B.25d", "glove.twitter.27B.50d", "glove.twitter.27B.100d",
        "glove.twitter.27B.200d":, "glove.6B.50d", "glove.6B.100d", "glove.6B.200d", "glove.6B.300d"

        :param random_embedding_dim: In the case of random embedding, this is the embedding dimension to use.

        """
        # call base constructor
        super(CLEVRDataset).__init__()

        # parse params
        self.set = set
        self.clevr_dir = clevr_dir
        self.clevr_humans = clevr_humans
        self.embedding_type = embedding_type
        self.random_embedding_dim = random_embedding_dim








if __name__ == '__main__':
    """ Unit test of CLEVRDataset"""
    set = 'train'
    clevr_dir = '/home/valbouy/CLEVR_v1.0'
    clevr_humans = False
    embedding_type = 'random'
    random_embedding_dim = 300

    clevr_dataset = CLEVRDataset(set, clevr_dir, clevr_humans, embedding_type, random_embedding_dim)
    print('Unit test completed.')
