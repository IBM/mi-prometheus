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

"""clevr.py: This file contains 1 class:

    - CLEVR, which represents the CLEVR `Dataset`. It inherits from \
    py:class:`miprometheus.problems.ImageTextToClassProblem`.

"""
__author__ = "Vincent Marois"

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import torch
import numpy as np
import os
import csv
import pickle
from PIL import Image

from torchvision import models
from torchvision import transforms

from miprometheus.utils.problems_utils.language import Language

from miprometheus.problems.image_text_to_class.image_text_to_class_problem import ImageTextToClassProblem


class CLEVR(ImageTextToClassProblem):
    """
    CLEVR Dataset class: Represents the CLEVR dataset.

    See reference here: https://cs.stanford.edu/people/jcjohns/clevr/

    :param params: Dictionary of parameters (read from configuration ``.yaml`` file).
    :type params: :py:class:`miprometheus.utils.ParamInterface`


    Given the relative complexity of this class, ``params`` should follow a specific template. Here are 2 examples:

        >>> params = {'settings': {'data_folder': '~/data/CLEVR_v1.0',
        >>>                        'set': 'train',
        >>>                        'dataset_variant': 'CLEVR'},
        >>>           'images': {'raw_images': False,
        >>>                      'feature_extractor': {'cnn_model': 'resnet101',
        >>>                                            'num_blocks': 4}},
        >>>           'questions': {'embedding_type': 'random', 'embedding_dim': 300,
        >>>                         'tasks': ['Exist']}})

        >>> params = {'settings': {'data_folder': '~/data/CLEVR_v1.0',
        >>>                        'set': 'train',
        >>>                        'dataset_variant': 'CLEVR-Humans'},
        >>>           'images': {'raw_images': True},
        >>>           'questions': {'embedding_type': 'glove.6B.300d',
        >>>                        'tasks': ['Exist', 'Count', 'CompareInteger', 'CompareAttribute', 'QueryAttribute']}}


    ``params`` is separated in 3 sections:

        - `settings`: generic settings for the :py:class:`CLEVR` class,
        - `images`: specific parameters for the images,
        - `questions`: specific parameters for the questions.


    Here is a breakdown of the available options:

        -   `settings`:

            - ``data_folder``: Root folder of the dataset. Will also be used to store `generated files/` \
               (e.g. tokenization of the questions, features extracted from the images etc.)

               .. warning::

                    As of now, this class doesn't handle downloading & decompressing the dataset if it is not \
                    present in the ``data_folder``. Please make sure that the dataset is already present in this \
                    ``data_folder``.

                        - For CLEVR-Humans, since only the questions change (and the images remains the same),\
                          please put the corresponding `.json` files in `~/CLEVR_v1.0/questions/`.
                        - For CLEVR-CoGenT, this is a fairly separate dataset with different questions & images.
                          Indicate ``data_folder`` as the root to `~/CLEVR_CoGenT_v1.0/` in this case.

            - ``set``: either "train", "val" in the case of "CLEVR" & "CLEVR-Humans", and "valA", "valB" or\
              "trainA" in the case of CLEVR-CoGenT. "test" is not supported yet since ground truth answers are\
              not distributed by the CLEVR authors.
            - ``dataset_variant``: either   "CLEVR", "CLEVR-CoGenT" or "CLEVR-Humans".

        - `images`:

            - ``raw_images``: whether or not to use to the original images as the visual source. If ``False``, then
              ``feature_extractor`` cannot be empty. The visual source will then be features extracted from the\
               original images using a specified pretrained CNN.
            - ``cnn_model`` : In the case of features extracted from the original images, the specific CNN model to\
            use. Must be part of :py:mod:`torchvision.models`.
            - ``num_blocks``: In the case of features extracted from the original images, this represents the number\
            of layers to use from ``cnn_model``.

                .. warning::

                    This is not verified in any way by this class.

        - `questions`:

            - ``embedding_type``: string to indicate the pretrained embedding to use: either "random" to use\
             ``nn.Embedding`` or one of the following:

                    - "charngram.100d",
                    - "fasttext.en.300d",
                    - "fasttext.simple.300d",
                    - "glove.42B.300d",
                    - "glove.840B.300d",
                    - "glove.twitter.27B.25d",
                    - "glove.twitter.27B.50d",
                    - "glove.twitter.27B.100d",
                    - "glove.twitter.27B.200d",
                    - "glove.6B.50d",
                    - "glove.6B.100d",
                    - "glove.6B.200d",
                    - "glove.6B.300d"

            - ``embedding_dim``: In the case of a random ``embedding_type``, this is the embedding dimension to use.
            - ``embedding_source``: In the case of a random ``embedding_type``, this is the source of the embeddings \
            to use. ``str``, equal to one of the dataset variant: "CLEVR", "CLEVR-CoGenT" or "CLEVR-Humans".

                .. warning::

                    If this ``embedding_source`` is different than the indicated ``dataset_variant`` above:

                        - The class assumes that there is exist in ``data_folder``/`generated_files`:

                            - A file `<embedding_source>_embedding_weights.pkl` corresponding to the random \
                            embedding weights to use,
                            - A file `<embedding_source>_dics.pkl` corresponding to the dicts ``{'words': index}`` & \
                            ``{'answer': index}``.

                        - The class will then override checking if the file containing the tokenized questions exist, \
                        and instead load the `<embedding_source>_dics.pkl` file, and use it to tokenize the questions.
                        - Nonetheless, the tokenized questions and dicts will **not** be saved to file.
                        - The class will also load the `<embedding_source>_embedding_weights.pkl` file and use it as \
                        the weights of the random embedding layer.

                    This is particularly useful to finetune or test a CLEVR-trained model on CoGenT-A or CoGenT-B.

                    `Should work for both the training & validation samples although only has been tested on validation \
                    samples so far`.

            - ``tasks``: A list of the questions categories to use. The processed (i.e tokenized) questions will be \
            added to the dataset if their `question_type` is present in this list. The 5 considered tasks are 'Exist', \
            'Count', 'CompareInteger', 'CompareAttribute' and 'QueryAttribute'.

                .. note::

                    The questions are filtered once they have been tokenized, and the necessary dicts (\
                    ``{'words': index}`` & ``{'answer': index}``) created. This is to ensures that all questions \
                    will share common tokens, irrespective of the considered tasks.


    .. note::

        The following is set by default:

        >>> params = {'settings': {'data_folder': '~/data/CLEVR_v1.0',
        >>>                        'set': 'train',
        >>>                        'dataset_variant': 'CLEVR'},
        >>>           'images': {'raw_images': True},
        >>>           'questions': {'embedding_type': 'random', 'embedding_dim': 300, 'embedding_source': 'CLEVR',
        >>>                        'tasks': ['Exist', 'Count', 'CompareInteger', 'CompareAttribute', 'QueryAttribute']}}


    """

    def __init__(self, params):
        """
        Instantiate the CLEVR class.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).
        :type params: :py:class:`miprometheus.utils.ParamInterface`


        """
        # Call base class constructors.
        super(CLEVR, self).__init__(params)

        # parse parameters from the params dict
        self.parse_param_tree(params)

        # define the default_values dict: holds parameters values that a model may need.
        self.default_values = {
            'num_classes': 28,
            'height':  480 if params['images']['raw_images'] else 14,
            'width':  320 if params['images']['raw_images'] else 14,
            'depth':  3 if params['images']['raw_images'] else 1024,
            'question_encoding_size': 300
            }

        # define the data_definitions dict: holds a description of the DataDict content
        self.data_definitions = {'images': {'size': [-1, 3, 480, 320] if params['images']['raw_images']
                                                                   else [-1, 1024, 14, 14],
                                            'type': [np.ndarray]},
                                 'questions': {'size': [-1, -1, -1], 'type': [torch.Tensor]},
                                 'questions_length': {'size': [-1], 'type': [list, int]},
                                 'questions_string': {'size': [-1, -1], 'type': [list, str]},
                                 'questions_type': {'size': [-1, -1], 'type': [list, str]},
                                 'targets': {'size': [-1], 'type': [torch.Tensor]},
                                 'targets_string': {'size': [-1, -1], 'type': [list, str]},
                                 'index': {'size': [-1], 'type': [list, int]},
                                 'imgfiles': {'size': [-1, -1], 'type': [list, str]}
                                 }

        # problem name
        self.name = 'CLEVR'

        self.logger.info('Loading the {} samples from {}'.format(self.set, self.dataset))

        # check if the folder /generated_files in self.data_folder already exists, if not create it:
        if not os.path.isdir(os.path.join(self.data_folder, 'generated_files')):
            self.logger.warning('Folder {} not found, creating it.'.format(os.path.join(self.data_folder,
                                                                                        'generated_files')))
            os.mkdir(os.path.join(self.data_folder, 'generated_files'))

        # check if the folder containing the images feature maps (processed by self.cnn_model) exists or not
        # For the same self.set, this file is the same for CLEVR & CLEVR-Humans
        # It will be different for CLEVR-CoGenT
        if not params['images']['raw_images']:
            if not os.path.isdir(os.path.join(self.data_folder, 'generated_files', self.cnn_model, self.set)):
                self.logger.warning('Directory {} not found on disk, extracting the features for each image and storing'
                                    ' them here.'.format(os.path.join(self.data_folder, 'generated_files', self.cnn_model, self.set)))
                self.generate_feature_maps_file()

        # check if the file containing the tokenized questions (& answers, image filename, type etc.) exists or not
        questions_filename = os.path.join(self.data_folder, 'generated_files', '{}_{}_questions.pkl'.format(self.set, self.dataset))
        if os.path.isfile(questions_filename) and self.embedding_source == self.dataset:
            self.logger.info('The file {} already exists, loading it.'.format(questions_filename))

            # load questions
            with open(questions_filename, 'rb') as questions:
                self.data_ = pickle.load(questions)

            # load word_dic & answer_dic
            with open(os.path.join(self.data_folder, 'generated_files', '{}_dics.pkl'.format(self.dataset)), 'rb') as f:
                dic = pickle.load(f)
                self.answer_dic = dic['answer_dic']
                self.word_dic = dic['word_dic']

        else:  # The file doesn't exist: Process the questions
            self.logger.warning('File {} not found on disk, processing the questions.'.format(questions_filename))

            # We need to ensure that we use the same words & answers dicts for both train & val, otherwise we do not
            # have the same reference.
            if self.set == 'val' or self.set == 'valA' or self.set == 'valB':  # handle CoGenT

                if self.embedding_source != self.dataset:
                    # load the specified dicts and re-tokenize the questions but don't save them to file.
                    with open(os.path.join(self.data_folder, 'generated_files', '{}_dics.pkl'.format(self.embedding_source)),
                              'rb') as f:
                        dic = pickle.load(f)
                        self.answer_dic = dic['answer_dic']
                        self.word_dic = dic['word_dic']
                        self.logger.info("Loaded the 'words': index & 'answer': index dicts from the "
                                         "embedding source '{}'.".format(self.embedding_source))

                    self.data_, self.word_dic, self.answer_dic = self.generate_questions_dics(self.set,
                                                                                              word_dic=self.word_dic,
                                                                                              answer_dic=self.answer_dic,
                                                                                              save_to_file=False)
                else:
                    # first generate the words dic using the training samples
                    self.logger.warning('We need to ensure that we use the same words-to-index & answers-to-index '
                                        'dictionaries for both the train & val samples.')
                    self.logger.warning('First, generating the words-to-index & answers-to-index dictionaries from '
                                        'the training samples :')
                    _, self.word_dic, self.answer_dic = self.generate_questions_dics('train' if self.set == 'val' else
                                                                                     'trainA',
                                                                                     word_dic=None,
                                                                                     answer_dic=None)

                    # then tokenize the questions using the created dictionaries from the training samples
                    self.logger.warning('We can now tokenize the validation questions using the dictionaries created from '
                                        'the training samples')
                    self.data_, self.word_dic, self.answer_dic = self.generate_questions_dics(self.set,
                                                                                             word_dic=self.word_dic,
                                                                                             answer_dic=self.answer_dic)

            elif self.set == 'train' or self.set == 'trainA':  # Can directly tokenize the questions
                if self.embedding_source != self.dataset:
                    # load the specified dicts and re-tokenize the questions but don't save them to file.
                    with open(os.path.join(self.data_folder, 'generated_files', '{}_dics.pkl'.format(self.embedding_source)),
                              'rb') as f:
                        dic = pickle.load(f)
                        self.answer_dic = dic['answer_dic']
                        self.word_dic = dic['word_dic']
                        self.logger.info("Loaded the 'words': index & 'answer': index dicts from the "
                                         "embedding source '{}'.".format(self.embedding_source))

                    self.data_, self.word_dic, self.answer_dic = self.generate_questions_dics(self.set,
                                                                                             word_dic=self.word_dic,
                                                                                             answer_dic=self.answer_dic,
                                                                                             save_to_file=False)
                else:
                    self.data_, self.word_dic, self.answer_dic = self.generate_questions_dics(self.set, word_dic=None,
                                                                                              answer_dic=None)

        # --> At this point, self.data_ contains the processed questions

        # create the objects for the specified embeddings
        if self.embedding_type == 'random':
            self.logger.info('Constructing random embeddings using a uniform distribution')
            # instantiate nn.Embeddings look-up-table with specified embedding_dim
            self.n_vocab = len(self.word_dic)+1
            self.embed_layer = torch.nn.Embedding(num_embeddings=self.n_vocab, embedding_dim=self.embedding_dim)

            # we have to make sure that the weights are the same during training and validation
            weights_filepath = os.path.join(self.data_folder, 'generated_files', '{}_embedding_weights.pkl'.format(self.embedding_source))
            if os.path.isfile(weights_filepath):
                self.logger.info('Found random embedding weights on file ({}), using them.'.format(weights_filepath))
                with open(weights_filepath, 'rb') as f:
                    self.embed_layer.weight.data = pickle.load(f)
            else:
                self.logger.warning('No weights found on file for random embeddings. Initializing them from a Uniform '
                                    'distribution and saving to file in {}'.format(weights_filepath))
                self.embed_layer.weight.data.uniform_(0, 1)
                with open(weights_filepath, 'wb') as f:
                    pickle.dump(self.embed_layer.weight.data, f)

        else:
            self.logger.info('Constructing embeddings using {}'.format(self.embedding_type))
            # instantiate Language class
            self.language = Language('lang')
            self.questions = [q['string_question'] for q in self.data]
            # use the questions set to construct the embeddings vectors
            self.language.build_pretrained_vocab(self.questions, vectors=self.embedding_type)

        # filter the qs based on their types (Exist, Count, CompareInteger, CompareAttribute, QueryAttribute)
        self.data = []
        for q in self.data_:
            if q['question_type'] in self.tasks:
                self.data.append(q)

        self.length = len(self.data)

        # Done! The actual question embedding is handled in __getitem__.

    def parse_param_tree(self, params):
        """
        Parses the parameters tree passed as input to the constructor.

        Due to the relative complexity inherent to the several variants of the `CLEVR` dataset (Humans, CoGenT)\
        and the processing available to both the images (features extraction or not) and the questions\
         (which type of embedding to use), this step is of relative importance.


        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).
        :type params: :py:class:`miprometheus.utils.ParamInterface`


        """
        # Set default parameters: load the original images & embed all questions randomly
        params.add_default_params({'settings': {'data_folder': '~/data/CLEVR_v1.0',
                                                'set': 'train',
                                                'dataset_variant': 'CLEVR'},
                                   'images': {'raw_images': 'True'},
                                   'questions': {'embedding_type': 'random', 'embedding_dim': 300,
                                                 'embedding_source': 'CLEVR',
                                                 'tasks': ['Exist', 'Count', 'CompareInteger', 'CompareAttribute',
                                                           'QueryAttribute']}
                                   })
        # get the data_folder
        self.data_folder = os.path.expanduser(params['settings']['data_folder'])

        # get the set descriptor
        self.set = params['settings']['set']
        assert self.set in ['train', 'val', 'test', 'trainA', 'valA', 'valB'], "self.set must be in" \
                                                     " ['train', 'val', 'test' 'trainA', 'valA', 'valB'], got {}".format(self.set)

        # We don't handle the creation of the test set for now since the ground truth answers are not distributed.
        if self.set == 'test':
            self.logger.error('Test set generation not supported for now since the ground truth answers '
                              'are not distributed. Exiting.')
            exit(0)

        # get the dataset variant
        self.dataset = params['settings']['dataset_variant']
        assert self.dataset in ['CLEVR', 'CLEVR-CoGenT', 'CLEVR-Humans'], "dataset_variant must be " \
                                                                          "in ['CLEVR', 'CLEVR-CoGenT', 'CLEVR-Humans'], got {}".format(
            self.dataset)

        if self.dataset == 'CLEVR' or self.dataset == 'CLEVR-Humans':
            assert 'CLEVR_v1.0' in self.data_folder, "Indicated data_folder does not contain 'CLEVR_v1.0'." \
                                                     "Please correct it. Got: {}".format(self.data_folder)
        elif self.dataset == 'CoGenT':
            assert 'CLEVR_CoGenT_v1.0' in self.data_folder, "Indicated data_folder does not contain " \
                                                            "'CLEVR_CoGenT_v1.0'.Please correct it." \
                                                            "Got: {}".format(self.data_folder)

        # get the images parameters:
        self.raw_image = params['images']['raw_images']
        if params['images']['raw_images']:
            self.image_source = os.path.join(self.data_folder, 'images', self.set)
        else:
            assert bool(params['images']['feature_extractor']), "The images source is either the original images " \
                                                                "or features extracted: Cannot have 'raw_images'= " \
                                                                "False and no parameters in 'feature_extractor'."
            # passed, so can continue parsing params
            self.cnn_model = params['images']['feature_extractor']['cnn_model']
            self.image_source = os.path.join(self.data_folder, 'generated_files', self.cnn_model, self.set)

            assert self.cnn_model in dir(models), "Did not find specified cnn_model in torchvision.models." \
                                                  " Available models: {}".format(dir(models))
            # this is too complex to check, not doing it.
            self.num_blocks = params['images']['feature_extractor']['num_blocks']

        # get the questions parameters:
        self.embedding_type = params['questions']['embedding_type']
        embedding_types = ["random", "charngram.100d", "fasttext.en.300d", "fasttext.simple.300d", "glove.42B.300d",
                           "glove.840B.300d", "glove.twitter.27B.25d", "glove.twitter.27B.50d",
                           "glove.twitter.27B.100d", "glove.twitter.27B.200d", "glove.6B.50d", "glove.6B.100d",
                           "glove.6B.200d", "glove.6B.300d"]

        assert self.embedding_type in embedding_types, "Embedding type not found, available options are {}".format(
            embedding_types)
        if self.embedding_type == 'random':
            self.embedding_dim = int(params['questions']['embedding_dim'])

            # checks if the embedding source is specified
            if 'embedding_source' in params['questions']:
                self.embedding_source = params['questions']['embedding_source']

                # checks if it is different than the dataset_variant
                if self.embedding_source != self.dataset:
                    self.logger.warning('Detected that the questions embedding source is different than the '
                                        'dataset variant. Got {} and the dataset variant is {}'.format(self.embedding_source,
                                                                                                       self.dataset))
                    self.logger.warning("Will override checking if the file containing the tokenized questions exist "
                                        "and re-tokenize the question using the {'words': index} & {'answer': index} "
                                        "dicts and random weights from the embedding source.")
            else:
                self.embedding_source = self.dataset

        else:
            self.embedding_dim = int(self.embedding_type[:-4])

        self.tasks = set(params['questions']['tasks'])
        possible_tasks = {'Exist', 'Count', 'CompareInteger', 'CompareAttribute', 'QueryAttribute'}
        assert self.tasks.issubset(possible_tasks), 'Unexpected tasks were detected. Available ones are {}, got {}.'\
            .format(possible_tasks, self.tasks)
        self.logger.info('Considering questions only in the {} categories.'.format(self.tasks))

    def generate_questions_dics(self, set, word_dic=None, answer_dic=None, save_to_file=True):
        """
        Loads the questions from the .json file, tokenize them, creates vocab dics and save that to files.

        :param set: String to specify which dataset to use: ``train``, ``val`` (``test`` not handled yet.)
        :type set: str

        :param word_dic: dict ``{'word': index}`` to be used to tokenize the questions. Optional. If passed, it\
        is used and unseen words are added. It not passed, an empty one is created.
        :type word_dic: dict

        :param answer_dic: dict ``{'answer': index}`` to be used to process the answers. Optional. If passed, it\
        is used and unseen answers are added. It not passed, an empty one is created.
        :type answer_dic: dict

        :param save_to_file: Whether to save to file the tokenized questions and the dicts.
        :type save_to_file: bool, default: True

        :return:

            - A dict, containing for each question:

                - The tokenized question,
                - The answer,
                - The original question string,
                - The original path to the associated image
                - The question type

            - The word_dic
            - The answer_dic

        """
        if word_dic is None:
            # create empty dic for the words vocab set
            word_dic = {}
        if answer_dic is None:
            # same for the answers
            answer_dic = {}

        import json
        import tqdm
        import nltk
        nltk.download('punkt')  # needed for nltk.word.tokenize

        # load questions from the .json file
        question_file = os.path.join(self.data_folder, 'questions', 'CLEVR-Humans-{}.json'.format(set) if self.dataset=='CLEVR-Humans' else 'CLEVR_{}_questions.json'.format(set))

        with open(question_file) as f:
            self.logger.info('Loading samples from {} ...'.format(question_file))
            data = json.load(f)
        self.logger.info('Loaded {} samples'.format(len(data['questions'])))

        # load the dict question_family_index -> task
        # If the file does not exist - create it.
        if not os.path.isfile(os.path.join(self.data_folder, 'generated_files/index_to_task.json')):
            index_to_task = '{"0": "CompareInteger", "1": "CompareInteger", "2": "CompareInteger", "3": "CompareInteger", "4": "CompareInteger", "5": "CompareInteger", "6": "CompareInteger", "7": "CompareInteger", "8": "CompareInteger", "9": "CompareAttribute", "10": "CompareAttribute", "11": "CompareAttribute", "12": "CompareAttribute", "13": "CompareAttribute", "14": "CompareAttribute", "15": "CompareAttribute", "16": "CompareAttribute", "17": "CompareAttribute", "18": "CompareAttribute", "19": "CompareAttribute", "20": "CompareAttribute", "21": "CompareAttribute", "22": "CompareAttribute", "23": "CompareAttribute", "24": "CompareAttribute", "25": "Count", "26": "Exist", "27": "QueryAttribute", "28": "QueryAttribute", "29": "QueryAttribute", "30": "QueryAttribute", "31": "Count", "32": "QueryAttribute", "33": "QueryAttribute", "34": "QueryAttribute", "35": "QueryAttribute", "36": "Exist", "37": "Exist", "38": "Exist", "39": "Exist", "40": "Count", "41": "Count", "42": "Count", "43": "Count", "44": "Exist", "45": "Exist", "46": "Exist", "47": "Exist", "48": "Count", "49": "Count", "50": "Count", "51": "Count", "52": "QueryAttribute", "53": "QueryAttribute", "54": "QueryAttribute", "55": "QueryAttribute", "56": "QueryAttribute", "57": "QueryAttribute", "58": "QueryAttribute", "59": "QueryAttribute", "60": "QueryAttribute", "61": "QueryAttribute", "62": "QueryAttribute", "63": "QueryAttribute", "64": "Count", "65": "Count", "66": "Count", "67": "Count", "68": "Count", "69": "Count", "70": "Count", "71": "Count", "72": "Count", "73": "Exist", "74": "QueryAttribute", "75": "QueryAttribute", "76": "QueryAttribute", "77": "QueryAttribute", "78": "Count", "79": "Exist", "80": "QueryAttribute", "81": "QueryAttribute", "82": "QueryAttribute", "83": "QueryAttribute", "84": "Count", "85": "Exist", "86": "QueryAttribute", "87": "QueryAttribute", "88": "QueryAttribute", "89": "QueryAttribute"}'
            with open(os.path.join(self.data_folder, 'generated_files/index_to_task.json'), 'w+') as f:
                f.write(index_to_task)
        # Load dict.
        with open(os.path.join(self.data_folder, 'generated_files/index_to_task.json')) as f:
            index_to_task = json.load(f)

        # start constructing vocab sets
        result = []
        word_index = 1  # 0 reserved for padding
        answer_index = 0

        self.logger.info('Constructing {} {} words dictionary:'.format(self.dataset, set))
        # progress bar
        t = tqdm.tqdm(total=len(data['questions']), unit=" questions", unit_scale=True, unit_divisor=1000)  # Initialize
        for question in data['questions']:
            words = nltk.word_tokenize(question['question'])
            question_token = []

            for word in words:
                token = word_dic.get(word, None)
                if not token:  # new word, add it to the dict
                    question_token.append(word_index)
                    word_dic[word] = word_index
                    word_index += 1
                else:
                    question_token.append(token)

            answer_word = question['answer']
            a_token = answer_dic.get(answer_word, None)
            if not a_token:
                answer = answer_index
                answer_dic[answer_word] = answer_index
                answer_index += 1
            else:
                answer = answer_dic[answer_word]

            # save sample params as a dict.
            question_type = index_to_task[str(question['question_family_index'])]

            result.append({'tokenized_question': question_token, 'answer': answer,
                           'string_question': question['question'], 'imgfile': question['image_filename'],
                           'question_type': question_type})
            t.update()
        t.close()

        self.logger.info('Done: constructed words dictionary of length {}, and answers dictionary of length {}'.format(len(word_dic),
                                                                                                            len(answer_dic)))
        if save_to_file:
            # save result to file
            questions_filename = os.path.join(self.data_folder, 'generated_files', '{}_{}_questions.pkl'.format(self.set, self.dataset))
            with open(questions_filename, 'wb') as f:
                pickle.dump(result, f)

            self.logger.warning('Saved tokenized questions to file {}.'.format(questions_filename))

            # save dictionaries to file:
            with open(os.path.join(self.data_folder, 'generated_files', '{}_dics.pkl'.format(self.dataset)), 'wb') as f:
                pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)
            self.logger.warning('Saved dics to file {}.'.format(os.path.join(self.data_folder, 'generated_files', '{}_dics.pkl'.format(self.dataset))))

        # return everything
        return result, word_dic, answer_dic

    def generate_feature_maps_file(self):
        """
        Uses :py:class:`miprometheus.utils.GenerateFeatureMaps` to pass the :py:class:`CLEVR` images through a \
        pretrained CNN model.

        """
        # import lines
        from miprometheus.utils.problems_utils.generate_feature_maps import GenerateFeatureMaps
        from torch.utils.data import DataLoader
        import tqdm

        # create DataLoader of the images dataset.
        dataset = GenerateFeatureMaps(image_dir=os.path.join(self.data_folder, 'images', self.set), set=self.set,
                                      cnn_model=self.cnn_model, num_blocks=self.num_blocks,
                                      transform=transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                         std=[0.229, 0.224, 0.225])]),
                                      filename_template='CLEVR_{}_{}.png'.format(self.set, '{}'))
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        size = len(dataloader)
        dataloader = iter(dataloader)
        pbar = tqdm.tqdm(dataloader, total=size, unit="images")

        # create the folder where the extracted features maps will be stored
        if not os.path.isdir(os.path.join(self.data_folder, 'generated_files', self.cnn_model, self.set)):
            os.makedirs(os.path.join(self.data_folder, 'generated_files', self.cnn_model, self.set))

        dir = os.path.join(self.data_folder, 'generated_files', self.cnn_model, self.set)

        with torch.no_grad():
            for i, image in enumerate(pbar):
                image = image.type(self.app_state.dtype)

                # forward pass, move output to cpu and store it into the file.
                features = dataset.model(image).detach().cpu().numpy()
                with open(os.path.join(dir, '{}_{}_{}.pt'.format('CLEVR-CoGenT' if self.dataset=='CLEVR-CoGenT' else 'CLEVR', self.set, str(i).zfill(6))), 'wb') as f:
                    torch.save(features, f)

        self.logger.warning('Features successfully extracted and stored in {}.'.format(dir))

    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a sample.

        :param index: index of the sample to return.
        :type index: int

        :return: DataDict({'images','questions', 'questions_length', 'questions_string', 'questions_type', 'targets', \
        'targets_string', 'index','imgfiles'}), with:

            - images: extracted feature maps from the raw image
            - questions: tensor of word indexes
            - questions_length: len(question)
            - questions_string: original question string
            - questions_type: category of the question (query, count...)
            - targets: index of the answer in the answers dictionary
            - targets_string: None for now
            - index: index of the sample
            - imgfiles: image filename

        """
        # load tokenized_question, answer, string_question, image_filename from self.data
        question, answer, question_string, imgfile, question_type = self.data[index].values()

        # create the image index to retrieve the feature maps or the original image
        index = str(imgfile.rsplit('_', 1)[1][:-4]).zfill(6)
        extension = '.png' if self.raw_image else '.pt'
        with open(os.path.join(self.image_source, '{}_{}_{}{}'.format('CLEVR-CoGenT' if self.dataset=='CLEVR-CoGenT' else 'CLEVR',
                                                                      self.set, index, extension)), 'rb') as f:
            if self.raw_image:
                img = Image.open(f).convert('RGB')  # for the original images
                img = transforms.ToTensor()(img).type(torch.FloatTensor).squeeze()
            else:
                img = torch.load(f)  # for feature maps
                img = torch.from_numpy(img).type(torch.FloatTensor).squeeze()

        # embed question
        if self.embedding_type == 'random':
            # embed question:
            question = self.embed_layer(torch.LongTensor(question)).type(torch.FloatTensor)

        else:
            # embed question
            question = self.language.embed_sentence(question_string)

        question_length = question.shape[0]

        # return everything
        data_dict = self.create_data_dict()

        data_dict['images'] = img
        data_dict['questions'] = question
        data_dict['questions_length'] = question_length
        data_dict['questions_string'] = question_string
        data_dict['questions_type'] = question_type
        data_dict['targets'] = answer
        # leave data_dict['target_string'] as None
        data_dict['index'] = index
        data_dict['imgfiles'] = imgfile

        return data_dict

    def collate_fn(self, batch):
        """
        Combines a list of DataDict (retrieved with :py:func:`__getitem__`) into a batch.

        .. note::

            Because each tokenized question has a variable length, padding is necessary to create batches.

            Hence, for a given batch, each question is padded to the length of the longest one.

            This length changes between batches, but this shouldn't be an issue.


        :param batch: list of individual samples to combine
        :type batch: list

        :return: DataDict({'images','questions', 'questions_length', 'questions_string', 'questions_type', 'targets', \
        'targets_string', 'index','imgfiles'})

        """
        batch_size = len(batch)

        # Get max question length, create tensor of shape [batch_size x maxQuestionLength x embedding]
        max_len = max(map(lambda x: x['questions_length'], batch))
        questions = torch.zeros(batch_size, max_len, self.embedding_dim).type(torch.FloatTensor)

        # create collecting data structures
        images = torch.zeros(batch_size, *batch[0]['images'].shape).type(torch.FloatTensor)
        targets = torch.zeros(batch_size).type(torch.FloatTensor)
        questions_length, questions_string, index, imgfiles, questions_type = [], [], [], [], []

        for idx, sample in enumerate(batch):
            qlen = sample['questions_length']

            images[idx] = sample['images']
            questions[idx, :qlen, :] = sample['questions']
            questions_length.append(qlen)
            questions_string.append(sample['questions_string'])
            questions_type.append(sample['questions_type'])
            targets[idx] = sample['targets']
            index.append(sample['index'])
            imgfiles.append(sample['imgfiles'])

        # construct the DataDict and fill it with the batch
        data_dict = self.create_data_dict()

        for key, val in [('images', images), ('questions', questions), ('questions_length', questions_length),
                         ('targets', targets), ('questions_string', questions_string), ('index', index),
                         ('imgfiles', imgfiles), ('questions_type', questions_type)]:
            data_dict[key] = val

        return data_dict

    def finalize_epoch(self, epoch):
        """
        Empty for now.

        Will call :py:func:`get_acc_per_family` to get the accuracy per family once it has been refactored.

        :param epoch: current epoch index
        :type epoch: int

        """
        # self.get_acc_per_family()

    def initialize_epoch(self, epoch):
        """
        Resets the accuracy per category counters.

        :param epoch: current epoch index
        :type epoch: int
        """

        self.categories_stats = dict(zip(self.categories.keys(), self.tuple_list))

    def get_acc_per_family(self, data_dict, logits):
        """
        Compute the accuracy per family for the current batch. Also accumulates
        the number of correct predictions & questions per family in self.correct_pred_families (saved
        to file).


        .. note::

            To refactor.


        :param data_dict: DataDict({'images','questions', 'questions_length', 'questions_string', 'questions_type', \
        'targets', 'targets_string', 'index','imgfiles'})
        :type data_dict: :py:class:`miprometheus.utils.DataDict`

        :param logits: network predictions.
        :type logits: :py:class:`torch.Tensor`

        """
        # unpack the DataDict
        question_types = data_dict['questions_type']
        targets = data_dict['targets']

        # get correct predictions
        pred = logits.max(1, keepdim=True)[1]
        correct = pred.eq(targets.view_as(pred))

        for i in range(correct.size(0)):
            # update # of questions for the corresponding family
            self.categories_stats[question_types[i]][1] += 1

            # update the # of correct predictions for the corresponding family
            if correct[i] == 1: self.categories_stats[question_types[i]][0] += 1

        categories_list = ['query_attribute', 'compare_integer', 'count', 'compare_attribute', 'exist']
        tuple_list_categories = [[0, 0] for _ in range(len(categories_list))]
        dic_categories = dict(zip(categories_list, tuple_list_categories))

        for category in categories_list:
            for family in self.categories.keys():
                if self.categories[family] == category:
                    dic_categories[category][0] += self.categories_stats[family][0]
                    dic_categories[category][1] += self.categories_stats[family][1]

        with open(os.path.join(self.data_folder, 'generated_files',
                               '{}_{}_categories_acc.csv'.format(self.dataset, self.set)), 'w') as f:
            writer = csv.writer(f)
        for key, value in self.categories_stats.items():
            writer.writerow([key, value])

    def show_sample(self, data_dict, sample=0):
        """

        Show a sample of the current DataDict.

        :param data_dict: DataDict({'images','questions', 'questions_length', 'questions_string', 'questions_type', 'targets', \
        'targets_string', 'index','imgfiles'})
        :type data_dict: :py:class:`miprometheus.utils.DataDict`

        :param sample: sample index to visualize.
        :type sample: int
        """
        # create plot figures
        plt.figure(1)

        # unpack data_dict
        questions_string = data_dict['questions_string']
        question_types = data_dict['questions_type']
        answers = data_dict['targets']
        imgfiles = data_dict['imgfiles']

        question = questions_string[sample]
        answer = answers[sample]
        answer = list(self.answer_dic.keys())[list(self.answer_dic.values()).index(answer.data)]  # dirty hack to go back from the
        # value in a dict to the key.

        # open image
        imgfile = imgfiles[sample]
        img = Image.open(os.path.join(self.data_folder, 'images', self.set, imgfile)).convert('RGB')
        img = np.array(img)

        plt.suptitle(question)
        plt.title('Question type: {}'.format(question_types[sample]))
        plt.xlabel('Answer: {}'.format(answer))
        plt.imshow(img)

        # show visualization
        plt.show()

    def plot_preprocessing(self, data_dict, logits):
        """
        Recover the predicted answer (as a string) from the logits and adds it to the current DataDict.
        Will be used in ``models.model.Model.plot()``.

        :param data_dict: DataDict({'images','questions', 'questions_length', 'questions_string', 'questions_type', 'targets', \
        'targets_string', 'index','imgfiles'})
        :type data_dict: :py:class:`miprometheus.utils.DataDict`

        :param logits: Predictions of the model.
        :type logits: :py:class:`torch.Tensor`

        :return:

            - data_dict with one added `predicted answer` key,
            - logits


        """

        # unpack data_dict
        answers = data_dict['targets']

        batch_size = logits.size(0)

        # get index of highest probability
        logits_indexes = torch.argmax(logits, dim=-1)

        prediction_string = [list(self.answer_dic.keys())[list(self.answer_dic.values()).index(
                logits_indexes[batch_num].data)] for batch_num in range(batch_size)]

        answer_string = [list(self.answer_dic.keys())[list(self.answer_dic.values()).index(
                answers[batch_num].data)] for batch_num in range(batch_size)]

        data_dict['targets_string'] = answer_string
        data_dict['predictions_string'] = prediction_string
        data_dict['clevr_dir'] = self.data_folder

        return data_dict, logits


if __name__ == "__main__":
    """Unit test that generates a batch and displays a sample."""

    from miprometheus.utils.param_interface import ParamInterface
    params = ParamInterface()
    params.add_config_params({'settings': {'data_folder': '/Users/vincentmarois/Downloads/CLEVR_CoGenT_v1.0',
                                           'set': 'trainA',
                                           'dataset_variant': 'CLEVR-CoGenT'},
                              'images': {'raw_images': False,
                                         'feature_extractor': {'cnn_model': 'resnet101',
                                                               'num_blocks': 4}},

                              'questions': {'embedding_type': 'random', 'embedding_dim': 300, 'embedding_source': 'CLEVR-CoGenT',
                                            'tasks': ['Exist', 'Count', 'CompareInteger', 'CompareAttribute', 'QueryAttribute']
                                            }})

    # create problem
    clevr_dataset = CLEVR(params)

    batch_size = 64

    sample = clevr_dataset[0]
    # print(repr(sample))
    print('__getitem__ works.')

    # instantiate DataLoader object
    problem = DataLoader(clevr_dataset, batch_size=batch_size, shuffle=False, collate_fn=clevr_dataset.collate_fn,
                         num_workers=0, sampler=None)

    import time
    s = time.time()
    for i, batch in enumerate(problem):
        print('Batch # {} - {}'.format(i, type(batch)))
        if i == 200:
            break

    print('Number of workers: {}'.format(problem.num_workers))
    print('time taken to generate 200 batches of size {}: {}s'.format(batch_size, time.time() - s))

    # Display single sample (0) from batch.
    # batch = next(iter(problem))
    # print(batch)
    #clevr_dataset.show_sample(batch, 0)

    print('Unit test completed.')
