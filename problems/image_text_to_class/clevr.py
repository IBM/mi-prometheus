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

"""clevr.py: This file contains 2 classes :

          - ClevrDataset (CLEVR object class): This class creates a Dataset object to represent a CLEVR dataset.
            ClevrDataset builds the embedding for the questions.

          - Clevr (CLEVR Problem class): This class generates batches over a CLEVRDataset object.
            It also has a show_sample() method that displays a sample (image, question, answer).
  """
__author__ = "Vincent Albouy"

import json
import numpy as np
import re
from PIL import Image
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

# Add path to main project directory
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..', '..'))

from problems.problem import DataTuple
from problems.image_text_to_class.image_text_to_class_problem import ImageTextToClassProblem, ImageTextTuple
from misc.app_state import AppState
from problems.utils.language import Language

app_state = AppState()


class ClevrDataset(Dataset):
    """
    Inherits from the Dataset class to represent the CLEVR dataset. Will be used by the Clevr class below to generate
    batches.

    ---------------
    class Dataset(object): An abstract class representing a Dataset.
    All other datasets should subclass it. All subclasses should override __len__, that provides the size of the dataset
    , and __getitem__, supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, train, clevr_dir, clevr_humans, embedding_type):
        """
        Instantiate a ClevrDataset object.

        :param train: Boolean to indicate whether or not the dataset is constructed for training.
        :param clevr_dir: Directory path to the CLEVR dataset.
        :param clevr_humans: Boolean to indicate whether to use the questions from CLEVR-Humans.
        :param embedding_type: string to indicate the pretrained embedding to use.

        """
        # call base constructor
        super(ClevrDataset).__init__()

        # parse params
        self.clevr_dir = clevr_dir
        self.clevr_humans = clevr_humans
        self.train = train
        self.embedding_type = embedding_type

        # instantiate Language class - This object will be used to create the words embeddings
        self.language = Language('lang')

        # load appropriate images & questions from CLEVR of CLEVR-Humans
        if train:
            self.quest_json_filename = os.path.join(self.clevr_dir, 'questions',
                                                    'CLEVR-Humans-train.json' if self.clevr_humans else 'CLEVR_train_questions.json')
            self.img_dir = os.path.join(self.clevr_dir, 'images', 'train')

        else:
            self.quest_json_filename = os.path.join(self.clevr_dir, 'questions',
                                                    'CLEVR-Humans-val.json' if self.clevr_humans else 'CLEVR_val_questions.json')
            self.img_dir = os.path.join(self.clevr_dir, 'images', 'val')

        # load questions
        with open(self.quest_json_filename, 'r') as json_file:
            self.questions = json.load(json_file)['questions']

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
        print(' ---> Constructing the dictionaries with word embedding, may take some time ')

        # making an empty list to store the vocabulary set of the dataset
        words_list = []

        # dictionary to contain the answers
        answ_to_ix = {}

        # load all words from the selected questions to words_list
        for q in tqdm(self.questions):  # tqdm displays a progress bar
            question = self.tokenize(q['question'])
            answer = q['answer']

            for word in question:
                if word not in words_list:
                    words_list.append(word)

            a = answer.lower()
            if a not in answ_to_ix:
                ix = len(answ_to_ix) + 1
                answ_to_ix[a] = ix

        # use the vocabulary set to construct the embeddings vectors
        print('Constructing the embeddings vectors for a vocabulary set of length {}'.format(len(words_list)))
        self.language.build_pretrained_vocab(words_list, vectors=self.embedding_type)

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
        return len(self.questions)

    def __getitem__(self, idx):
        """
        Getter method to access the dataset and return a sample.

        :param idx: index of the sample to return.

        :return: {'image': image, 'question': question, 'answer': answer, 'current_question': current_question}
        """
        # get current question with indices idx
        current_question = self.questions[idx]

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


class Clevr(ImageTextToClassProblem):
    """CLEVR Problem class: This class generates batches over a ClevrDataset object.
        It also has a show_sample method that displays a sample (image, question, answer)"""

    def __init__(self, params):
        """
        Instantiate the Clevr class.

        :param params: dict of parameters.
        """

        # parse parameters from the params dict
        self.clevr_dir = params['CLEVR_dir']
        self.train = params['train']
        self.batch_size = params['batch_size']
        self.clevr_humans = params['clevr_humans']
        self.embedding_type = params['embedding_type']
        self.clevr_dataset = ClevrDataset(self.train, self.clevr_dir, self.clevr_humans, self.embedding_type)

    def generate_batch(self):
        """
        Generates a batch from self.clevr_dataset.

        :return: - data_tuple: ((images, questions), answers)
                 - aux_tuple: (questions_strings, answers_strings)
        """

        clevr_train_loader = DataLoader(self.clevr_dataset, batch_size=self.batch_size,
                                        sampler=RandomSampler(self.clevr_dataset))
        clevr_train_loader = next(iter(clevr_train_loader))

        images = clevr_train_loader['image']
        questions = clevr_train_loader['question']
        answers = clevr_train_loader['answer']
        current_questions = clevr_train_loader['current_question']
        aux_tuple = (current_questions['question'], current_questions['answer'])
        image_text_tuple = ImageTextTuple(images, questions)

        return DataTuple(image_text_tuple, answers), aux_tuple

    def show_sample(self, data_tuple, aux_tuple, sample_number=0):
        """

        :param data_tuple: DataTuple: ((images, questions), answers)
        :param aux_tuple: AuxTuple: (questions_strings, answers_strings)
        :param sample_number: sample index to visualize.
        """
        # create plot figures
        plt.figure(1)

        # unpack tuples
        ((image, question), answer) = data_tuple
        (written_question, written_answer) = aux_tuple

        plt.title(written_question[sample_number])
        plt.xlabel(written_answer[sample_number])

        image = image[sample_number, :, :, :]
        plt.imshow(image)

        # show visualization
        plt.show()


if __name__ == "__main__":
    """Unit test that generates a batch and displays a sample."""

    params = {'batch_size': 5, 'CLEVR_dir': 'CLEVR_v1.0', 'train': True, 'clevr_humans': False,
              'embedding_type': 'glove.6B.100d'}

    # create problem
    problem = Clevr(params)

    # generate a batch
    data_tuple, aux_tuple = problem.generate_batch()

    print(data_tuple)
    print(aux_tuple)

    problem.show_sample(data_tuple, aux_tuple, sample_number=2)
