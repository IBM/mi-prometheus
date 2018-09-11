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

          - Clevr (CLEVR Problem class): This class generates batches over a CLEVRDataset object.
            It also has a show_sample() method that displays a sample (image, question, answer).
  """
__author__ = "Vincent Albouy, Vincent Marois"

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import torch
import csv
import os
import h5py
import pickle

from problems.utils.language import Language
from problems.problem import DataDict

from problems.image_text_to_class.image_text_to_class_problem import ImageTextToClassProblem
from misc.app_state import AppState
app_state = AppState()

import logging
logger = logging.getLogger('CLEVR')


class CLEVR(ImageTextToClassProblem):
    """CLEVR Problem class: This class generates batches over a CLEVRDataset object.
        It also has a show_sample method that displays a sample (image, question, answer)"""

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

        # to compute the accuracy per family
        self.family_list = ['query_size', 'equal_size', 'query_shape', 'query_color', 'greater_than', 'equal_material',
                            'equal_color', 'equal_shape', 'less_than', 'count', 'exist', 'equal_integer', 'query_material']
        self.tuple_list = [[0, 0] for _ in range(len(self.family_list))]
        self.dic = dict(zip(self.family_list, self.tuple_list))
        self.categories_transform = {'query_size': 'query_attribute', 'equal_size': 'compare_attribute',
                                     'query_shape': 'query_attribute', 'query_color': 'query_attribute',
                                     'greater_than': 'compare_integer', 'equal_material': 'compare_attribute',
                                     'equal_color': 'compare_attribute', 'equal_shape': 'compare_attribute',
                                     'less_than': 'compare_integer', 'count': 'count', 'exist': 'exist',
                                     'equal_integer': 'compare_integer', 'query_material': 'query_attribute'}

        self.data_definition = {'img': {'size': [320, 480, 3], 'type': torch.Tensor},
                                'question': {'size': 'var', 'type': int},
                                'question_length': {'size': 1, 'type': int},
                                'targets': {'size': 1, 'type': str},
                                'string_question': {'size': 1, 'type': str},
                                'index': {'size': 1, 'type': str},
                                'imgfile': {'size': 1, 'type': str},
                                'question_type': {'size': 1, 'type': str}}


        if self.set == 'test':
            logger.error('Test set generation not supported for now. Exiting.')
            exit(0)

        logger.info('Loading the {} samples from {}'.format(set, 'CLEVR-Humans' if self.clevr_humans else 'CLEVR'))

        # check if the folder /generated_files in self.clevr already exists, if not creates it:
        if not os.path.isdir(self.clevr_dir + '/generated_files'):
            logger.warning('Folder {} not found, creating it.'.format(self.clevr_dir + '/generated_files'))
            os.mkdir(self.clevr_dir + '/generated_files')

        # checking if the file containing the images feature maps (processed by ResNet101) exists or not
        # For the same self.set, this file is the same for CLEVR & CLEVR-Humans
        feature_maps_filename = self.clevr_dir + '/generated_files/{}_CLEVR_features.hdf5'.format(self.set)
        if os.path.isfile(feature_maps_filename):
            logger.info('The file {} already exists, loading it.'.format(feature_maps_filename))

        else:
            logger.warning('File {} not found on disk, generating it:'.format(feature_maps_filename))
            self.generate_feature_maps_file(feature_maps_filename)

        # actually load the file
        self.h = h5py.File(feature_maps_filename, 'r')
        self.img = self.h['data']

        # checking if the file containing the tokenized questions (& answers, image filename) exists or not
        questions_filename = self.clevr_dir + '/generated_files/{}_{}_questions.pkl'.format(self.set, 'CLEVR_Humans' if self.clevr_humans else 'CLEVR')
        if os.path.isfile(questions_filename):
            logger.info('The file {} already exists, loading it.'.format(questions_filename))

            # load questions
            with open(questions_filename, 'rb') as questions:
                self.data = pickle.load(questions)

            # load word_dics & answer_dics
            with open(self.clevr_dir + '/generated_files/dics.pkl', 'rb') as f:
                dic = pickle.load(f)
                self.answer_dic = dic['answer_dic']
                self.word_dic = dic['word_dic']

        else:
            logger.warning('File {} not found on disk, generating it.'.format(questions_filename))

            # WARNING: We need to ensure that we use the same words & answers dics for both train & val, otherwise we
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
                self.data, self.word_dic, self.answer_dic = self.generate_questions_dics(self.set, word_dic=self.word_dic,
                                                                                         answer_dic=self.answer_dic)

            elif self.set == 'train' or self.set == 'trainA':  # self.set=='train', we can directly tokenize the questions
                self.data, self.word_dic, self.answer_dic = self.generate_questions_dics(self.set, word_dic=None, answer_dic=None)

        # At this point, the objects self.img & self.data contains the feature maps & questions

        # creates the objects for the specified embeddings
        if self.embedding_type == 'random':
            logger.info('Constructing random embeddings using a uniform distribution')
            # instantiate nn.Embeddings look-up-table with specified embedding_dim
            self.n_vocab = len(self.word_dic)+1
            self.embed_layer = torch.nn.Embedding(num_embeddings=self.n_vocab, embedding_dim=self.random_embedding_dim)

            # we have to make sure that the weights are the same during train or val!
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

    def get_acc_per_family(self, data_tuple, aux_tuple, logits):
        """
        Compute the accuracy per family for the current batch. Also accumulates the # of correct predictions & questions
        per family in self.dic (saved to file).

        :param data_tuple: DataTuple ((images, questions), targets)
        :param aux_tuple: (questions_strings, questions_indexes, images_filenames, question_types)
        :param logits: network predictions.
        """

        # get correct predictions
        pred = logits.max(1, keepdim=True)[1]
        correct = pred.eq(data_tuple.targets.view_as(pred))

        # unpack aux_tuple
        (s_questions, indexes, imgfiles, question_types) = aux_tuple
        print('\n')
        for i in range(correct.size(0)):
            # update # of questions for the corresponding family
            self.dic[question_types[i]][1] += 1

            if correct[i] == 1:
                # update the # of correct predictions for the corresponding family
                self.dic[question_types[i]][0] += 1

        for family in self.family_list:
            if self.dic[family][1] == 0:
                print('Family: {} - Acc: No questions!'.format(family))

            else:
                family_accuracy = (self.dic[family][0]) / (self.dic[family][1])
                print('Family: {} - Acc: {} - Total # of questions: {}'.format(family, family_accuracy, self.dic[family][1]))

        categories_list = ['query_attribute', 'compare_integer', 'count', 'compare_attribute', 'exist']
        tuple_list_categories = [[0, 0] for _ in range(len(categories_list))]
        dic_categories = dict(zip(categories_list, tuple_list_categories))
        print('\n')
        for category in categories_list:
            for family in self.family_list:
                if self.categories_transform[family] == category:
                    dic_categories[category][0] += self.dic[family][0]
                    dic_categories[category][1] += self.dic[family][1]

        for category in categories_list:
            if dic_categories[category][1] == 0:
                print('Category: {} - Acc: No questions!'.format(category))

            else:
                category_accuracy = (dic_categories[category][0]) / (dic_categories[category][1])
                print('Category: {} - Acc: {} - Total # of questions: {}'.format(category, category_accuracy, dic_categories[category][1]))

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

        # load the dictionary question_family_type -> question_type: Will allow to plot the accuracy per question category
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

        :return: img: extracted feature maps from the raw image
                 tokenized_question: tensor of word indexes
                 len(question): question length
                 answer: index of the answer in the answers dictionary
                 string_question: original question string
                 index: index of the sample
                 imgfile: image filename
        """
        # load tokenized_question, answer, string_question, image_filename from self.data
        question, answer, string_question, imgfile, question_type = self.data[index].values()

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
        data_dict = DataDict({key: None for key in self.data_definition.keys()})

        data_dict['img'] = img
        data_dict['question'] = question
        data_dict['question_length'] = question_length
        data_dict['targets'] = answer
        data_dict['string_question'] = string_question
        data_dict['index'] = index
        data_dict['imgfile'] = imgfile
        data_dict['question_type'] = question_type

        return data_dict

    def collate_data(self, batch):
        """
        Combines samples (retrieved with __getitem__) into a mini-batch
        :param batch: list (?) of samples to combine

        :return: images (tensor), padded_tokenized_questions (tensor), questions_lengths (list), answers (tensor),
                questions_strings (list), indexes (list), imgfiles (list)
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
            image, question, length, answer, string_question, index, imgfile, question_type = b.values()

            images.append(image)
            lengths.append(length)
            answers.append(answer)
            s_questions.append(string_question)
            indexes.append(index)
            imgfiles.append(imgfile)
            question_types.append(question_type)

            questions[i, :length, :] = question

        data_dict = DataDict({key: None for key in self.data_definition.keys()})
        data_dict['img'] = torch.stack(images).type(app_state.dtype)
        data_dict['question'] = questions
        data_dict['question_length'] = lengths
        data_dict['targets'] = torch.tensor(answers).type(app_state.LongTensor)
        data_dict['string_question'] = s_questions
        data_dict['index'] = indexes
        data_dict['imgfile'] = imgfiles
        data_dict['question_type'] = question_types

        return data_dict

    def collect_statistics(self, stat_col, data_tuple, logits):
        """
        Collects accuracy.
        :param stat_col: Statistics collector.
        :param data_tuple: Data tuple containing inputs and targets.
        :param logits: Logits being output of the model.
        :param _: auxiliary tuple (aux_tuple) is not used in this function. 
        """
        stat_col['acc'] = self.calculate_accuracy(data_tuple, logits)


        #self.get_acc_per_family(data_tuple, aux_tuple, logits)

    def generate_batch(self):
        """
        Generates a batch from self.clevr_dataset.

        WARNING: WE PASS THE QUESTIONS LENGTH INTO THE DATATUPLE!

        :return: - data_tuple: (((images, questions), questions_len), answers)
                 - aux_tuple: (questions_strings, questions_indexes, images_filenames, question_types) (visualization)
        """

        clevr_loader = DataLoader(self.clevr_dataset, batch_size=self.batch_size, collate_fn=self.clevr_dataset.collate_data,
                                  sampler=RandomSampler(self.clevr_dataset))

        images, questions, questions_len, answers, s_questions, indexes, imgfiles, question_types = next(iter(clevr_loader))

        # create data_tuple
        image_text_tuple = ImageTextTuple(images, questions)
        inner_data_tuple = (image_text_tuple, questions_len)
        data_tuple = DataTuple(inner_data_tuple, answers)

        aux_tuple = (s_questions, indexes, imgfiles, question_types)

        return data_tuple, aux_tuple

    def turn_on_cuda(self, data_dict, aux_tuple):
        """
        Enables computations on GPU - copies the input and target matrices (from DataTuple) to GPU.

        :param data_tuple: Data tuple.
        :param aux_tuple: Auxiliary tuple (WARNING: Values stored in that variable will remain on CPU)
        :returns: Pair of Data and Auxiliary tuples (Data on GPU, Aux on CPU).
        """

        return data_dict.cuda(), aux_tuple

    def show_sample(self, data_tuple, aux_tuple, sample_number=0):
        """
        :param data_tuple: DataTuple: ((images, questions), answers)
        :param aux_tuple: AuxTuple: (questions_strings, answers_strings)
        :param sample_number: sample index to visualize.
        """
        # create plot figures
        plt.figure(1)

        # unpack aux_tuple
        (s_questions, indexes, imgfiles, question_types) = aux_tuple

        question = s_questions[sample_number]
        answer = answers[sample_number]
        answer = list(self.clevr_dataset.answer_dic.keys())[list(self.clevr_dataset.answer_dic.values()).index(answer.data)]  # dirty hack to go back from the
        # value in a dict to the key.
        imgfile = imgfiles[sample_number]

        from PIL import Image
        import numpy
        img = Image.open(self.clevr_dir + '/images/' + self.set + '/' + imgfile).convert('RGB')
        img = numpy.array(img)
        plt.suptitle(question)
        plt.title('Question type: {}'.format(question_types[sample_number]))
        plt.xlabel('Answer: {}'.format(answer))
        plt.imshow(img)

        # show visualization
        plt.show()

    def plot_preprocessing(self, data_tuple, aux_tuple, logits):
        """
        Allows for some data preprocessing before the model creates a plot for visualization during training or
        inference.
        To be redefined in inheriting classes.
        :param data_tuple: Data tuple.
        :param aux_tuple: Auxiliary tuple.
        :param logits: Logits being output of the model.
        :return: data_tuplem aux_tuple, logits after preprocessing.
        """

        # unpack data_tuple
        inner_tuple, answer = data_tuple
        image_questions_tuple, questions_len = inner_tuple

        batch_size = logits.size(0)

        # get index of highest probability
        logits_indexes = torch.argmax(logits, dim=-1)

        prediction_string = [list(self.clevr_dataset.answer_dic.keys())[list(self.clevr_dataset.answer_dic.values()).index(logits_indexes[batch_num].data)] for
                          batch_num in range(batch_size)]
        answer_string = [list(self.clevr_dataset.answer_dic.keys())[list(self.clevr_dataset.answer_dic.values()).index(answer[batch_num].data)] for
                          batch_num in range(batch_size)]

        (s_questions, indexes, imgfiles, question_types) = aux_tuple
        aux_tuple = (s_questions, answer_string, imgfiles, self.set, prediction_string, self.clevr_dir)

        return aux_tuple, data_tuple, logits


if __name__ == "__main__":
    """Unit test that generates a batch and displays a sample."""

    params = {'batch_size': 64, 'CLEVR_dir': '/home/vmarois/Downloads/CLEVR_v1.0', 'set': 'train', 'clevr_humans': False,
              'embedding_type': 'random', 'random_embedding_dim': 300}

    # create problem
    clevr_dataset = CLEVR(params)
    sample = clevr_dataset[0]
    print('__getitem__ works.')

    # instantiate DataLoader object
    problem = DataLoader(clevr_dataset, batch_size=params['batch_size'], shuffle=False,
                         collate_fn=clevr_dataset.collate_data)

    # generate a batch
    for i_batch, sample in enumerate(problem):
        print('Sample # {} - {}'.format(i_batch, sample['img'].shape), type(sample))
        break

    print('Unit test completed.')
