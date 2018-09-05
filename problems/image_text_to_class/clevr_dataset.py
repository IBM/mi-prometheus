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

    def __init__(self, set, clevr_dir, clevr_humans,
                 embedding_type='random', random_embedding_dim=300):
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

        if self.set == 'test':
            logger.error('Test set generation not supported for now. Exiting.')
            exit(0)

        logger.info('Loading the {} samples from {}'.format(
            set, 'CLEVR-Humans' if self.clevr_humans else 'CLEVR'))

        # check if the folder /generated_files in self.clevr already exists, if
        # not creates it:
        if not os.path.isdir(self.clevr_dir + '/generated_files'):
            logger.warning('Folder {} not found, creating it.'.format(
                self.clevr_dir + '/generated_files'))
            os.mkdir(self.clevr_dir + '/generated_files')

        # checking if the file containing the images feature maps (processed by ResNet101) exists or not
        # For the same self.set, this file is the same for CLEVR & CLEVR-Humans
        feature_maps_filename = self.clevr_dir + \
            '/generated_files/{}_CLEVR_features.hdf5'.format(self.set)
        if os.path.isfile(feature_maps_filename):
            logger.info('The file {} already exists, loading it.'.format(
                feature_maps_filename))

        else:
            logger.warning('File {} not found on disk, generating it:'.format(
                feature_maps_filename))
            self.generate_feature_maps_file(feature_maps_filename)

        # actually load the file
        self.h = h5py.File(feature_maps_filename, 'r')
        self.img = self.h['data']

        # checking if the file containing the tokenized questions (& answers,
        # image filename) exists or not
        questions_filename = self.clevr_dir + '/generated_files/{}_{}_questions.pkl'.format(
            self.set, 'CLEVR_Humans' if self.clevr_humans else 'CLEVR')
        if os.path.isfile(questions_filename):
            logger.info('The file {} already exists, loading it.'.format(
                questions_filename))

            # load questions
            with open(questions_filename, 'rb') as questions:
                self.data = pickle.load(questions)

            # load word_dics & answer_dics
            with open(self.clevr_dir + '/generated_files/dics.pkl', 'rb') as f:
                dic = pickle.load(f)
                self.answer_dic = dic['answer_dic']
                self.word_dic = dic['word_dic']

        else:
            logger.warning(
                'File {} not found on disk, generating it.'.format(questions_filename))

            # WARNING: We need to ensure that we use the same words & answers dics for both train & val, otherwise we
            # do not have the same reference!
            if self.set == 'val' or self.set == 'valA' or self.set == 'valB':
                # first generate the words dic using the training samples
                logger.warning(
                    'We need to ensure that we use the same words-to-index & answers-to-index dictionaries '
                    'for both the train & val samples.')
                logger.warning(
                    'First, generating the words-to-index & answers-to-index dictionaries from '
                    'the training samples :')
                _, self.word_dic, self.answer_dic = self.generate_questions_dics(
                    'train' if self.set == 'val' else 'trainA', word_dic=None, answer_dic=None)

                # then tokenize the questions using the created dictionaries
                # from the training samples
                logger.warning(
                    'Then we can tokenize the validation questions using the dictionaries '
                    'created from the training samples')
                self.data, self.word_dic, self.answer_dic = self.generate_questions_dics(
                    self.set, word_dic=self.word_dic, answer_dic=self.answer_dic)

            # self.set=='train', we can directly tokenize the questions
            elif self.set == 'train' or self.set == 'trainA':
                self.data, self.word_dic, self.answer_dic = self.generate_questions_dics(
                    self.set, word_dic=None, answer_dic=None)

        # At this point, the objects self.img & self.data contains the feature
        # maps & questions

        # creates the objects for the specified embeddings
        if self.embedding_type == 'random':
            logger.info(
                'Constructing random embeddings using a uniform distribution')
            # instantiate nn.Embeddings look-up-table with specified
            # embedding_dim
            self.n_vocab = len(self.word_dic) + 1
            self.embed_layer = torch.nn.Embedding(
                num_embeddings=self.n_vocab,
                embedding_dim=self.random_embedding_dim)

            # we have to make sure that the weights are the same during train
            # or val!
            if os.path.isfile(self.clevr_dir +
                              '/generated_files/random_embedding_weights.pkl'):
                logger.info(
                    'Found random embedding weights on file, using them.')
                with open(self.clevr_dir + '/generated_files/random_embedding_weights.pkl', 'rb') as f:
                    self.embed_layer.weight.data = pickle.load(f)
            else:
                logger.warning(
                    'No weights found on file for random embeddings. Initializing them from a Uniform '
                    'distribution and saving to file in {}'.format(
                        self.clevr_dir + '/generated_files/random_embedding_weights.pkl'))
                self.embed_layer.weight.data.uniform_(0, 1)
                with open(self.clevr_dir + '/generated_files/random_embedding_weights.pkl', 'wb') as f:
                    pickle.dump(self.embed_layer.weight.data, f)

        else:
            logger.info('Constructing embeddings using {}'.format(
                self.embedding_type))
            # instantiate Language class
            self.language = Language('lang')
            self.questions = [q['string_question'] for q in self.data]
            # use the questions set to construct the embeddings vectors
            self.language.build_pretrained_vocab(
                self.questions, vectors=self.embedding_type)

        # Done! The actual question embedding is handled in __getitem__.

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
        question_file = os.path.join(
            self.clevr_dir,
            'questions',
            'CLEVR-Humans-{}.json'.format(set) if self.clevr_humans else 'CLEVR_{}_questions.json'.format(set))
        with open(question_file) as f:
            logger.info('Loading samples from {} ...'.format(question_file))
            data = json.load(f)
        logger.info('Loaded {} samples'.format(len(data['questions'])))

        # load the dictionary question_family_type -> question_type: Will allow
        # to plot the accuracy per question category
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

                except BaseException:
                    question_token.append(word_index)
                    word_dic[word] = word_index
                    word_index += 1

            answer_word = question['answer']
            try:
                answer = answer_dic[answer_word]

            except BaseException:
                answer = answer_index
                answer_dic[answer_word] = answer_index
                answer_index += 1

            # save sample params as a dict.
            result.append({'tokenized_question': question_token, 'answer': answer,
                           'string_question': question['question'], 'imgfile': question['image_filename'],
                           'question_type': index_to_family[str(question['question_family_index'])]})

        logger.info(
            'Done: constructed words dictionary of length {}, and answers dictionary of length {}'.format(
                len(word_dic), len(answer_dic)))
        # save result to file
        questions_filename = self.clevr_dir + '/generated_files/{}_{}_questions.pkl'.format(
            set, 'CLEVR_Humans' if self.clevr_humans else 'CLEVR')
        with open(questions_filename, 'wb') as f:
            pickle.dump(result, f)

        logger.warning(
            'Saved tokenized questions to file {}.'.format(questions_filename))

        # save dictionaries to file:
        with open(self.clevr_dir + '/generated_files/dics.pkl', 'wb') as f:
            pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)
        logger.warning('Saved dics to file {}.'.format(
            self.clevr_dir + '/generated_files/dics.pkl'))

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
        generate_feature_maps = GenerateFeatureMaps(
            clevr_dir=self.clevr_dir,
            set=self.set,
            cnn_model='resnet101',
            num_blocks=4)
        dataloader = DataLoader(generate_feature_maps, batch_size=batch_size)

        size = len(dataloader)
        dataset = iter(dataloader)
        pbar = tqdm.tqdm(dataset, total=size, unit="batches")

        # create file to store feature maps.
        f = h5py.File(feature_maps_filename, 'w', libver='latest')
        dset = f.create_dataset(
            'data', (size * batch_size, 1024, 14, 14), dtype='f4')

        with torch.no_grad():
            for i, image in enumerate(pbar):
                # move batch to GPU
                image = image.type(torch.cuda.FloatTensor)

                # forward pass, move output to cpu and store it into the file.
                features = generate_feature_maps.model(
                    image).detach().cpu().numpy()
                dset[i * batch_size:(i + 1) * batch_size] = features

        f.close()
        logger.warning('File {} successfully created.'.format(
            feature_maps_filename))

    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a sample.

        :param index: index of the sample to return.

        :return: img: extracted feature maps from the raw image
                 tokenized_question: tensor of word indexes
                 len(question): question length
                 answer: index of the answer in the answers dictionary
                 string_question: original question string
                 index: index of the sample
                 imgfile: image filename
        """
        # load tokenized_question, answer, string_question, image_filename from
        # self.data
        question, answer, string_question, imgfile, question_type = self.data[index].values(
        )

        # create the image index to retrieve the feature maps in self.img
        id = int(imgfile.rsplit('_', 1)[1][:-4])

        img = torch.from_numpy(self.img[id]).type(app_state.dtype)

        # embed question
        if self.embedding_type == 'random':
            # embed question:
            question = self.embed_layer(
                torch.LongTensor(question)).type(app_state.dtype)

        else:
            # embed question
            question = self.language.embed_sentence(string_question)

        question_length = question.shape[0]

        # return everything
        return img, question, question_length, answer, string_question, index, imgfile, question_type

    def collate_data(self, batch):
        """
        Combines samples (retrieved with __getitem__) into a mini-batch
        :param batch: list (?) of samples to combine

        :return: images (tensor), padded_tokenized_questions (tensor), questions_lengths (list), answers (tensor),
                questions_strings (list), indexes (list), imgfiles (list)
        """
        # create list placeholders
        images, lengths, answers, s_questions, indexes, imgfiles, question_types = [
        ], [], [], [], [], [], []
        batch_size = len(batch)

        # get max question length, create tensor of shape [batch_size x maxQuestionLength] & sort questions by
        # decreasing length
        max_len = max(map(lambda x: len(x[1]), batch))
        sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

        # create tensor containing the embedded questions
        if self.embedding_type == 'random':
            questions = torch.zeros(
                batch_size,
                max_len,
                self.random_embedding_dim).type(
                app_state.dtype)

        else:
            # get embedding dimension from the embedding type
            embedding_dim = int(self.embedding_type[-4:-1])
            questions = torch.zeros(
                batch_size, max_len, embedding_dim).type(app_state.dtype)

        # fill in the placeholders
        for i, b in enumerate(sort_by_len):
            image, question, length, answer, string_question, index, imgfile, question_type = b

            images.append(image)
            lengths.append(length)
            answers.append(answer)
            s_questions.append(string_question)
            indexes.append(index)
            imgfiles.append(imgfile)
            question_types.append(question_type)

            questions[i, :length, :] = question

        # return all
        return torch.stack(images).type(
            app_state.dtype), questions, lengths, torch.tensor(answers).type(
            app_state.LongTensor), s_questions, indexes, imgfiles, question_types


if __name__ == '__main__':
    """ Unit test of CLEVRDataset"""
    set = 'train'
    clevr_dir = '/home/valbouy/CLEVR_v1.0'
    clevr_humans = False
    embedding_type = 'random'
    random_embedding_dim = 300

    clevr_dataset = CLEVRDataset(
        set, clevr_dir, clevr_humans, embedding_type, random_embedding_dim)
    print('Unit test completed.')
