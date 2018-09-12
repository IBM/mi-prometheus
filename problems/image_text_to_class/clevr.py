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

    - CLEVR, which represents the CLEVR `Dataset()`. It inherits from `ImageTextToClassProblem`.

"""
__author__ = "Vincent Marois, Vincent Albouy"

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import torch
import csv
import os
import h5py
import pickle

from utils.language import Language
from problems.problem import DataDict

from problems.image_text_to_class.image_text_to_class_problem import ImageTextToClassProblem
from utils.app_state import AppState
app_state = AppState()

import logging
logger = logging.getLogger('CLEVR')

import numpy as np

class CLEVR(ImageTextToClassProblem):
    """
    CLEVR Dataset class: Represents the CLEVR dataset.

    See reference here: https://cs.stanford.edu/people/jcjohns/clevr/

    .. note::

        - This class mainly checks if the files containing the features maps extracted from the images (via a CNN) and
          tokenized questions already exist. If not, it generates them for the specified sub-set.
        - `self.img` contains then the extracted feature maps.
        - `self.data` contains the tokenized questions, the associated image filenames, the answers, \
           the question strings and the question types.
        - The questions are then embedded based on the specified embedding. This embedding is random by default, but
          pretrained ones are possible.

    :param set: String to specify which dataset to use: 'train', 'val' or 'test'.
    :type set: str

    :param clevr_dir: Directory path to the CLEVR_v1.0 dataset. Will also be used to store the generated files \
    (.hdf5, .pkl)
    :type clevr_dir: str

    :param clevr_humans: Boolean to indicate whether to use the questions from CLEVR-Humans.
    :type clevr_humans: bool

    :param embedding_type: string to indicate the pretrained embedding to use: either 'random' to use nn.Embedding \
    or one of the following:

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


    :type embedding_type: str

    :param random_embedding_dim: In the case of random embedding, this is the embedding dimension to use.
    :type random_embedding_dim: int



    .. warning::

        As of now, this class doesn't handle downloading & decompressing the dataset to a specific folder.
        It assumes that the entire dataset is located in `self.clevr_dir`, which is a path to a directory that
        you can pass as a param.


    .. warning::

        **As of now, this class doesn't handle accessing the raw images as the visual input. It defaults to
        the file containing the extracted feature maps. The reason for that is that this class was implemented
        mainly for the MAC network at the beginning, which doesn't use the raw images but the feature maps extracted
        by `ResNet101.`**
        **It is planned to add support for using the original images as the visual input in the future.**


    """

    def __init__(self, params):
        """
        Instantiate the CLEVR class.

        :param params: dict of parameters.

        """
        # Call base class constructors.
        super(CLEVR, self).__init__(params)

        # parse parameters from the params dict
        self.clevr_dir = params['CLEVR_dir']
        self.set = params['set']
        self.batch_size = params['batch_size']
        self.clevr_humans = params['clevr_humans']
        self.embedding_type = params['embedding_type']
        self.random_embedding_dim = params['random_embedding_dim']

        # define the data_definitions dict: holds a description of the DataDict content
        self.default_values = {'num_inputs': 8 }

        # define the data_definitions dict: holds a description of the DataDict content
        self.data_definitions = {'img': {'size': [-1, 320, 480, 3], 'type': np.ndarray},
                                 'question': {'size': [-1, -1], 'type': torch.Tensor},
                                 'question_length': {'size': [-1], 'type': [list, int]},
                                 'question_string': {'size': [-1,-1], 'type': [list, str]},
                                 'question_type': {'size': [-1,-1], 'type': [list, str]},
                                 'targets': {'size': [-1], 'type': torch.Tensor},
                                 'targets_string': {'size': [-1,-1], 'type': [list, str]},
                                 'index': {'size': [-1], 'type': [list, int]},
                                 'imgfile': {'size': [-1,-1], 'type': [list,str]}
                                 }

        '''
        class MyModel(Model):
            def __init__(self, params, problem_default_values_ = {}):

            # Process all params from view and default_values_ here...

            # define the data_definitions dict: holds a description of the DataDict content
            self.data_definitions = {'img': {'size': [320, 480, 3], 'type': 'numpy.ndarray'} }

            self.data_definitions["img"]['size'] = [self.params['width'],self.params['height'], default_values_['num_inputs']]


            def handshake_definitions(self, data_definitions_):
                # do the handshake
        '''



        # the sub-types of question families
        self.family_list = [
            'query_size',
            'equal_size',
            'query_shape',
            'query_color',
            'greater_than',
            'equal_material',
            'equal_color',
            'equal_shape',
            'less_than',
            'count',
            'exist',
            'equal_integer',
            'query_material']

        # for storing the number of correct predictions & total number of questions per family
        self.tuple_list = [[0, 0] for _ in range(len(self.family_list))]
        self.dic = dict(zip(self.family_list, self.tuple_list))

        # link a sub-family to one of the main 5 categories of questions
        self.categories_transform = {
            'query_size': 'query_attribute',
            'equal_size': 'compare_attribute',
            'query_shape': 'query_attribute',
            'query_color': 'query_attribute',
            'greater_than': 'compare_integer',
            'equal_material': 'compare_attribute',
            'equal_color': 'compare_attribute',
            'equal_shape': 'compare_attribute',
            'less_than': 'compare_integer',
            'count': 'count',
            'exist': 'exist',
            'equal_integer': 'compare_integer',
            'query_material': 'query_attribute'}

        # We don't handle the creation of the test set for now since the ground truth answers are not distributed.
        if self.set == 'test':
            logger.error('Test set generation not supported for now. Exiting.')
            exit(0)

        logger.info('Loading the {} samples from {}'.format(set, 'CLEVR-Humans' if self.clevr_humans else 'CLEVR'))

        # check if the folder /generated_files in self.clevr_dir already exists, if not create it:
        if not os.path.isdir(self.clevr_dir + '/generated_files'):
            logger.warning('Folder {} not found, creating it.'.format(self.clevr_dir + '/generated_files'))
            os.mkdir(self.clevr_dir + '/generated_files')

        # check if the file containing the images feature maps (processed by ResNet101) exists or not
        # For the same self.set, this file is the same for CLEVR & CLEVR-Humans
        # TODO: We will have to separate this file into 1 file per sample to support multiprocessing!
        feature_maps_filename = self.clevr_dir + '/generated_files/{}_CLEVR_features.hdf5'.format(self.set)
        if os.path.isfile(feature_maps_filename):
            logger.info('The file {} already exists, loading it.'.format(feature_maps_filename))

        else:
            logger.warning('File {} not found on disk, generating it:'.format(feature_maps_filename))
            self.generate_feature_maps_file(feature_maps_filename)

        # load the file
        self.h = h5py.File(feature_maps_filename, 'r')
        self.img = self.h['data']

        # check if the file containing the tokenized questions (& answers, image filename, type etc.) exists or not
        questions_filename = self.clevr_dir + '/generated_files/{}_{}_questions.pkl'.format(self.set, 'CLEVR_Humans' if
                                                                                        self.clevr_humans else 'CLEVR')
        if os.path.isfile(questions_filename):
            logger.info('The file {} already exists, loading it.'.format(questions_filename))

            # load questions
            with open(questions_filename, 'rb') as questions:
                self.data = pickle.load(questions)

            # load word_dic & answer_dic
            with open(self.clevr_dir + '/generated_files/dics.pkl', 'rb') as f:
                dic = pickle.load(f)
                self.answer_dic = dic['answer_dic']
                self.word_dic = dic['word_dic']

        else:  # The file doesn't exist: Process the questions
            logger.warning('File {} not found on disk, generating it.'.format(questions_filename))

            # WARNING: We need to ensure that we use the same words & answers dicts for both train & val, otherwise we
            # do not have the same reference!
            if self.set == 'val' or self.set == 'valA' or self.set == 'valB':
                # first generate the words dic using the training samples
                logger.warning('We need to ensure that we use the same words-to-index & answers-to-index dictionaries '
                               'for both the train & val samples.')
                logger.warning('First, generating the words-to-index & answers-to-index dictionaries from '
                               'the training samples :')
                _, self.word_dic, self.answer_dic = self.generate_questions_dics('train' if self.set == 'val' else 'trainA',
                                                                                 word_dic=None, answer_dic=None)

                # then tokenize the questions using the created dictionaries from the training samples
                logger.warning('Then we can tokenize the validation questions using the dictionaries '
                               'created from the training samples')
                self.data, self.word_dic, self.answer_dic = self.generate_questions_dics(self.set,
                                                                                         word_dic=self.word_dic,
                                                                                         answer_dic=self.answer_dic)

            elif self.set == 'train' or self.set == 'trainA':  # self.set=='train', we can directly tokenize the questions
                self.data, self.word_dic, self.answer_dic = self.generate_questions_dics(self.set, word_dic=None, answer_dic=None)

        # --> At this point, the objects self.img & self.data contains the feature maps & questions

        # create the objects for the specified embeddings
        if self.embedding_type == 'random':
            logger.info('Constructing random embeddings using a uniform distribution')
            # instantiate nn.Embeddings look-up-table with specified embedding_dim
            self.n_vocab = len(self.word_dic)+1
            self.embed_layer = torch.nn.Embedding(num_embeddings=self.n_vocab, embedding_dim=self.random_embedding_dim)

            # we have to make sure that the weights are the same during training and validation!
            if os.path.isfile(self.clevr_dir + '/generated_files/random_embedding_weights.pkl'):
                logger.info('Found random embedding weights on file, using them.')
                with open(self.clevr_dir + '/generated_files/random_embedding_weights.pkl', 'rb') as f:
                    self.embed_layer.weight.data = pickle.load(f)
            else:
                logger.warning('No weights found on file for random embeddings. Initializing them from a Uniform '
                               'distribution and saving to file in {}'.format(self.clevr_dir +
                                                                              '/generated_files/random_embedding_weights.pkl'))
                self.embed_layer.weight.data.uniform_(0, 1)
                with open(self.clevr_dir + '/generated_files/random_embedding_weights.pkl', 'wb') as f:
                    pickle.dump(self.embed_layer.weight.data, f)

        else:
            logger.info('Constructing embeddings using {}'.format(self.embedding_type))
            # instantiate Language class
            self.language = Language('lang')
            self.questions = [q['string_question'] for q in self.data]
            # use the questions set to construct the embeddings vectors
            self.language.build_pretrained_vocab(self.questions, vectors=self.embedding_type)

        # Done! The actual question embedding is handled in __getitem__.

    def get_acc_per_family(self, data_dict, logits):
        """
        Compute the accuracy per family for the current batch. Also accumulates
        the number of correct predictions & questions per family in self.correct_pred_families (saved
        to file).

        TODO: This function needs refactoring to:
            - Only print the accuracy per family at the end of the epoch
            - Better code design -> use statistics aggregators.
        :param data_dict: DataDict {'img','question', 'question_length', 'targets', 'string_question', 'index', \
        'imgfile', 'question_type'}

        :param logits: network predictions.

        """
        # unpack the DataDict
        img, question, question_length, targets, string_question, index, imgfile, question_types = data_dict.values()

        # get correct predictions
        pred = logits.max(1, keepdim=True)[1]
        correct = pred.eq(targets.view_as(pred))

        for i in range(correct.size(0)):
            # update # of questions for the corresponding family
            self.dic[question_types[i]][1] += 1

            if correct[i] == 1:
                # update the # of correct predictions for the corresponding
                # family
                self.dic[question_types[i]][0] += 1

        for family in self.family_list:
            if self.dic[family][1] == 0:
                print('Family: {} - Acc: No questions!'.format(family))

            else:
                family_accuracy = (self.dic[family][0]) / (self.dic[family][1])
                print('Family: {} - Acc: {} - Total # of questions: {}'.format(family,
                                                                               family_accuracy, self.dic[family][1]))

        categories_list = ['query_attribute', 'compare_integer',
                           'count', 'compare_attribute', 'exist']
        tuple_list_categories = [[0, 0] for _ in range(len(categories_list))]
        dic_categories = dict(zip(categories_list, tuple_list_categories))

        for category in categories_list:
            for family in self.family_list:
                if self.categories_transform[family] == category:
                    dic_categories[category][0] += self.dic[family][0]
                    dic_categories[category][1] += self.dic[family][1]

        for category in categories_list:
            if dic_categories[category][1] == 0:
                print('Category: {} - Acc: No questions!'.format(category))

            else:
                category_accuracy = (
                    dic_categories[category][0]) / (dic_categories[category][1])
                print('Category: {} - Acc: {} - Total # of questions: {}'.format(
                    category, category_accuracy, dic_categories[category][1]))

        with open(self.clevr_dir + '/generated_files/families_acc.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in self.dic.items():
                writer.writerow([key, value])

    def __len__(self):
        """Return the length of the questions set"""
        return len(self.data)

    def close(self):
        """Close hdf5 file."""
        self.h.close()

    def generate_questions_dics(self, set, word_dic=None, answer_dic=None):
        """
        Loads the questions from the .json file, tokenize them, creates vocab dics and save that to files.

        :param set: String to specify which dataset to use: 'train', 'val'
        :param word_dic: dict {'word': index} to be used to tokenize the questions
        :param answer_dic: dict {'answer': index} to be used to process questions
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
        question_file = os.path.join(self.clevr_dir, 'questions', 'CLEVR-Humans-{}.json'.format(set) if self.clevr_humans
                                            else 'CLEVR_{}_questions.json'.format(set))
        with open(question_file) as f:
            logger.info('Loading samples from {} ...'.format(question_file))
            data = json.load(f)
        logger.info('Loaded {} samples'.format(len(data['questions'])))

        # load the dict question_family_type -> question_type: Will allow to plot the acc per question category
        with open(os.path.join(self.clevr_dir, 'questions/index_to_family.json')) as f:
            index_to_family = json.load(f)

        # start constructing vocab sets
        result = []
        word_index = 1  # 0 reserved for padding
        answer_index = 0

        logger.info('Constructing {} words dictionary:'.format(set))
        for question in tqdm.tqdm(data['questions']):
            words = nltk.word_tokenize(question['question'])
            question_token = []

            for word in words:
                try:
                    question_token.append(word_dic[word])

                except:
                    question_token.append(word_index)
                    word_dic[word] = word_index
                    word_index += 1

            answer_word = question['answer']
            try:
                answer = answer_dic[answer_word]

            except:
                answer = answer_index
                answer_dic[answer_word] = answer_index
                answer_index += 1

            # save sample params as a dict.
            result.append({'tokenized_question': question_token, 'answer': answer,
                           'string_question': question['question'], 'imgfile': question['image_filename'],
                           'question_type': index_to_family[str(question['question_family_index'])]})

        logger.info('Done: constructed words dictionary of length {}, and answers dictionary of length {}'.format(len(word_dic),
                                                                                                            len(answer_dic)))
        # save result to file
        questions_filename = self.clevr_dir + '/generated_files/{}_{}_questions.pkl'.format(set,
                                                                                            'CLEVR_Humans' if self.clevr_humans else 'CLEVR')
        with open(questions_filename, 'wb') as f:
            pickle.dump(result, f)

        logger.warning('Saved tokenized questions to file {}.'.format(questions_filename))

        # save dictionaries to file:
        with open(self.clevr_dir + '/generated_files/dics.pkl', 'wb') as f:
            pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)
        logger.warning('Saved dics to file {}.'.format(self.clevr_dir + '/generated_files/dics.pkl'))

        # return everything
        return result, word_dic, answer_dic

    def generate_feature_maps_file(self, feature_maps_filename, batch_size=50):
        """
        Uses GenerateFeatureMaps to pass the CLEVR images through a pretrained CNN model.

        :param set: String to specify which dataset to use: 'train', 'val' or 'test'.
        :param feature_maps_filename: filename for saving to file.
        :param batch_size: batch size

        :return: feature maps
        """
        # import lines
        from problems.image_text_to_class.generate_feature_maps import GenerateFeatureMaps
        from torch.utils.data import DataLoader
        import tqdm

        # create DataLoader of the images dataset.
        generate_feature_maps = GenerateFeatureMaps(clevr_dir=self.clevr_dir, set=self.set, cnn_model='resnet101', num_blocks=4)
        dataloader = DataLoader(generate_feature_maps, batch_size=batch_size)

        size = len(dataloader)
        dataset = iter(dataloader)
        pbar = tqdm.tqdm(dataset, total=size, unit="batches")

        # create file to store feature maps.
        f = h5py.File(feature_maps_filename, 'w', libver='latest')
        dset = f.create_dataset('data', (size * batch_size, 1024, 14, 14), dtype='f4')

        with torch.no_grad():
            for i, image in enumerate(pbar):
                # move batch to GPU
                image = image.type(torch.cuda.FloatTensor)

                # forward pass, move output to cpu and store it into the file.
                features = generate_feature_maps.model(image).detach().cpu().numpy()
                dset[i * batch_size:(i + 1) * batch_size] = features

        f.close()
        logger.warning('File {} successfully created.'.format(feature_maps_filename))

    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a sample.

        :param index: index of the sample to return.

        :return: DataDict({'img','question', 'question_length', 'question_string', 'question_type', 'targets', \
        'targets_string', 'index','imgfile'}), with:

            - img: extracted feature maps from the raw image
            - question: tensor of word indexes
            - question_length: len(question)
            - question_string: original question string
            - question_type: category of the question (query, count...)
            - targets: index of the answer in the answers dictionary
            - targets_string: None for now
            - index: index of the sample
            - imgfile: image filename


        .. warning::

            This function does not yet support multiprocessing for faster data loading as the feature maps are all
            stored in one unique file. **It is planned to separate them into one file per image in the future.**


        """
        # load tokenized_question, answer, string_question, image_filename from self.data
        question, answer, question_string, imgfile, question_type = self.data[index].values()

        # create the image index to retrieve the feature maps in self.img
        id = int(imgfile.rsplit('_', 1)[1][:-4])

        img = torch.from_numpy(self.img[id]).type(app_state.dtype)

        # embed question
        if self.embedding_type == 'random':
            # embed question:
            question = self.embed_layer(torch.LongTensor(question)).type(app_state.dtype)

        else:
            # embed question
            question = self.language.embed_sentence(string_question)

        question_length = question.shape[0]

        # return everything
        data_dict = DataDict({key: None for key in self.data_definitions.keys()})

        data_dict['img'] = img
        data_dict['question'] = question
        data_dict['question_length'] = question_length
        data_dict['question_string'] = question_string
        data_dict['question_type'] = question_type
        data_dict['targets'] = answer

        data_dict['index'] = index
        data_dict['imgfile'] = imgfile

        return data_dict

    def collate_fn(self, batch):
        """
        Combines a list of DataDict (retrieved with __getitem__) into a batch.

        .. note::

            Because each tokenized question has a variable length, padding is necessary.
            This means that the default collate_function does not work.

            **There has to be a better way to not loop over each sample in the list!!**

        :param batch: list of individual samples to combine

        :return: DataDict({'img','question', 'question_length', 'question_string', 'question_type', 'targets', \
        'targets_string', 'index','imgfile'})

        """
        # create list placeholders
        images, lengths, answers, s_questions, indexes, imgfiles, question_types = [], [], [], [], [], [], []
        batch_size = len(batch)

        # get max question length, create tensor of shape [batch_size x maxQuestionLength] & sort questions by
        # decreasing length
        max_len = max(map(lambda x: x['question_length'], batch))
        sort_by_len = sorted(batch, key=lambda x: x['question_length'], reverse=True)

        # create tensor containing the embedded questions
        if self.embedding_type == 'random':
            questions = torch.zeros(batch_size, max_len, self.random_embedding_dim).type(app_state.dtype)

        else:
            # get embedding dimension from the embedding type
            embedding_dim = int(self.embedding_type[-4:-1])
            questions = torch.zeros(batch_size, max_len, embedding_dim).type(app_state.dtype)

        # fill in the placeholders
        for i, b in enumerate(sort_by_len):
            image, question, question_length, question_string, question_type, answer, _, index, imgfile = b.values()

            images.append(image)
            lengths.append(question_length)
            answers.append(answer)
            s_questions.append(question_string)
            indexes.append(index)
            imgfiles.append(imgfile)
            question_types.append(question_type)

            questions[i, :question_length, :] = question

        # construct the DataDict and fill it with the batch
        data_dict = DataDict({key: None for key in self.data_definitions.keys()})

        data_dict['img'] = torch.stack(images).type(app_state.dtype)
        data_dict['question'] = questions
        data_dict['question_length'] = lengths
        data_dict['targets'] = torch.tensor(answers).type(app_state.LongTensor)
        data_dict['question_string'] = s_questions
        data_dict['index'] = indexes
        data_dict['imgfile'] = imgfiles
        data_dict['question_type'] = question_types

        return data_dict

    # set self.collate_fn to this new function

    def collect_statistics(self, stat_col, data_tuple, logits):
        """
        Collects accuracy.

        :param stat_col: Statistics collector.
        :param data_tuple: Data tuple containing inputs and targets.
        :param logits: Logits being output of the model.

        :param _: auxiliary tuple (aux_tuple) is not used in this function. 
        """
        stat_col['acc'] = self.calculate_accuracy(data_tuple, logits)

        # self.get_acc_per_family(data_tuple, aux_tuple, logits)

    def finalize_epoch(self):
        """
        Call `self.get_acc_per_family()` to get the accuracy per family.

        """
        #self.get_acc_per_family()

    def get_epoch_size(self):
        """
        :return: number of episodes to run to cover the entire chosen set once.
        """
        return self.__len__() // self.batch_size

    def initialize_epoch(self):
        """
        Resets the accuracy per family counters.

        """
        # self.reset_acc_per_family()  # TODO: Actually implement this function...

    def show_sample(self, data_dict, sample_number=0):
        """
        Show a sample of the current DataDict.
        :param data_dict: DataDict({'img','question', 'question_length', 'targets', 'string_question', 'index', \
        'imgfile', 'question_type'})

        :param sample_number: sample index to visualize.
        """
        # create plot figures
        plt.figure(1)

        # unpack data_dict
        images, questions, questions_len, questions_string, question_types, answers, _, indexes, imgfiles = data_dict.values()

        question = questions_string[sample_number]
        answer = answers[sample_number]
        answer = list(self.answer_dic.keys())[
            list(self.answer_dic.values()).index(answer.data)]  # dirty hack to go back from the
        # value in a dict to the key.
        imgfile = imgfiles[sample_number]

        from PIL import Image
        import numpy
        img = Image.open(self.clevr_dir + '/images/' +
                         self.set + '/' + imgfile).convert('RGB')
        img = numpy.array(img)
        plt.suptitle(question)
        plt.title('Question type: {}'.format(question_types[sample_number]))
        plt.xlabel('Answer: {}'.format(answer))
        plt.imshow(img)

        # show visualization
        plt.show()

    def plot_preprocessing(self, data_dict, logits):
        """
        Recover the predicted answer (as a string) from the logits.
        Will be used in model.plot()

        :param data_tuple: Data tuple.
        :param logits: Logits being output of the model.

        :return: data_tuple, aux_tuple, logits after preprocessing.

        """

        # unpack data_dict
        images, questions, questions_len, questions_string, question_types, answers, _, indexes, imgfiles = data_dict.values()

        batch_size = logits.size(0)

        # get index of highest probability
        logits_indexes = torch.argmax(logits, dim=-1)

        prediction_string = [list(self.answer_dic.keys())[list(self.answer_dic.values()).index(
                logits_indexes[batch_num].data)] for batch_num in range(batch_size)]

        answer_string = [list(self.clevr_dataset.answer_dic.keys())[list(self.clevr_dataset.answer_dic.values()).index(
                answers[batch_num].data)] for batch_num in range(batch_size)]

        # TODO: Here, we should be able to add these 2 new lists to DataDict so that they can be used in model.plot().

        new_data_dict = dict(data_dict.items() + {'prediction_string': prediction_string, 'answer_string':answer_string})

        return data_dict, logits


if __name__ == "__main__":
    """Unit test that generates a batch and displays a sample."""

    from utils.param_interface import ParamInterface
    params = ParamInterface()
    params.add_default_params({'batch_size': 64,
                               'CLEVR_dir': '/home/vmarois/Downloads/CLEVR_v1.0',
                               # 'data_folder': '~/data/CLEVR_v1.0/', # TODO!
                               'set': 'train',
                               'clevr_humans': False,
                               'embedding_type': 'random',
                               'random_embedding_dim': 300})

    # create problem
    clevr_dataset = CLEVR(params)
    print('Number of episodes to run to cover the set once: {}'.format(clevr_dataset.get_epoch_size()))

    sample = clevr_dataset[0]
    print('__getitem__ works.')

    # instantiate DataLoader object
    problem = DataLoader(clevr_dataset, batch_size=params['batch_size'], shuffle=True,
                         collate_fn=clevr_dataset.collate_fn)

    # generate a batch
    for i_batch, sample in enumerate(problem):
        print('Sample # {} - {}'.format(i_batch, sample['img'].shape), type(sample))
        # try to show a sample
        clevr_dataset.show_sample(data_dict=sample)
        break

    print('Unit test completed.')
