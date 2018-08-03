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

"""clevr_dataset.py: This file contains 1 class:

          - ClevrDataset (CLEVR object class): This class creates a Dataset object to represent a CLEVR dataset.
            ClevrDataset builds the embedding for the questions.
  """
__author__ = "Vincent Albouy, Vincent Marois"

import json
import numpy as np
import re
from PIL import Image
import torch
from tqdm import tqdm

from torch.utils.data import Dataset

# Add path to main project directory
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..', '..'))
from problems.utils.language import Language


class ClevrDataset(Dataset):
    """
    Inherits from the Dataset class to represent the CLEVR dataset. Will be used by the Clevr class below to generate
    batches.

    ---------------
    class Dataset(object): An abstract class representing a Dataset.
    All other datasets should subclass it. All subclasses should override __len__, that provides the size of the dataset
    , and __getitem__, supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, set, clevr_dir, clevr_humans, embedding_type):
        """
        Instantiate a ClevrDataset object.

        :param set: String to specify which dataset to use: 'train', 'val' or 'test'.
        :param clevr_dir: Directory path to the CLEVR dataset.
        :param clevr_humans: Boolean to indicate whether to use the questions from CLEVR-Humans.
        :param embedding_type: string to indicate the pretrained embedding to use.

        """
        # call base constructor
        super(ClevrDataset).__init__()

        # parse params
        self.clevr_dir = clevr_dir
        self.clevr_humans = clevr_humans
        self.set = set
        self.embedding_type = embedding_type

        # instantiate Language class - This object will be used to create the words embeddings
        self.language = Language('lang')

        # load appropriate images & questions from CLEVR or CLEVR-Humans
        if set == 'train':
            self.quest_json_filename = os.path.join(self.clevr_dir, 'questions',
                                                    'CLEVR-Humans-train.json' if self.clevr_humans else 'CLEVR_train_questions.json')
            self.img_dir = os.path.join(self.clevr_dir, 'images', 'train')

        elif set == 'val':
            self.quest_json_filename = os.path.join(self.clevr_dir, 'questions',
                                                    'CLEVR-Humans-val.json' if self.clevr_humans else 'CLEVR_val_questions.json')
            self.img_dir = os.path.join(self.clevr_dir, 'images', 'val')

        else:
            self.quest_json_filename = os.path.join(self.clevr_dir, 'questions',
                                                    'CLEVR-Humans-test.json' if self.clevr_humans else 'CLEVR_test_questions.json')
            self.img_dir = os.path.join(self.clevr_dir, 'images', 'test')

        # load samples
        with open(self.quest_json_filename, 'r') as json_file:
            print('Loading samples from {}'.format(self.quest_json_filename))
            self.samples = json.load(json_file)['questions']
            print('Loading done')

        # build the embeddings
        self.answ_to_ix = self.build_embeddings()

    def tokenize(self, sentence):
        """
        Separate the punctuations from the words to tokenize it afterwards.
        Also lowercase the sentence.

        :param sentence: sentence to process

        :return: 'tokenized' sentence.
        """

        # punctuation should be separated from the words
        s = re.sub('([.,;:!?()])', r' \1 ', sentence)
        s = re.sub('\s{2,}', ' ', s)

        # tokenize
        split = s.split()

        # normalize all words to lowercase
        lower = [w.lower() for w in split]

        return lower

    def build_embeddings(self):
        """Creates the word embeddings.

        - 1. Builds the vocabulary set of the loaded questions
        - 2. Uses Language object to create the embeddings

         If it is the first time you run this code, it will take some time to load the embeddings.

         :return the answers to index dictionary.
         """
        # making an empty list to store the questions
        questions = []

        # dictionary to contain the answers
        answ_to_ix = {}

        # load all words from the selected questions to words_list
        print('Selecting questions & answers:')
        for q in tqdm(self.samples):  # tqdm displays a progress bar
            questions.append(q['question'])
            answer = q['answer']

            a = answer.lower()
            if a not in answ_to_ix:
                ix = len(answ_to_ix) + 1
                answ_to_ix[a] = ix

        # use the questions set to construct the embeddings vectors
        print('Constructing embeddings using {}'.format(self.embedding_type))
        self.language.build_pretrained_vocab(questions, vectors=self.embedding_type)

        return answ_to_ix

    def to_dictionary_indexes(self, dictionary, sentence):
        """
        Outputs indexes of the dictionary corresponding to the words in the sequence.
        Case insensitive.
        """
        split = self.tokenize(sentence)
        idxs = torch.tensor([dictionary[w] for w in split])

        return idxs

    def __len__(self):
        """Return the length of the questions set"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Getter method to access the dataset and return a sample.

        :param idx: index of the sample to return.

        :return: {'image': image, 'question': question, 'answer': answer, 'current_question': current_question}
        """
        # get current sample with indices idx
        current_question = self.samples[idx]

        # load image and convert it to RGB format
        img_filename = os.path.join(self.img_dir, current_question['image_filename'])
        image = Image.open(img_filename).convert('RGB')
        image = np.array(image)

        # choose padding size
        padding_size = 50

        # embed the question
        question = self.language.embed_sentence(current_question['question'])

        # padding the question
        padding = torch.zeros((padding_size - question.size(0), question.size(1)))
        question = torch.cat((question, padding), 0)

        # get answer index
        answer = self.to_dictionary_indexes(self.answ_to_ix, current_question['answer'])

        # make a dictionary with all the outputs
        sample = {'image': image, 'question': question, 'answer': answer, 'current_question': current_question}

        return sample
