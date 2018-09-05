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

from problems.problem import DataTuple

from problems.image_text_to_class.image_text_to_class_problem import ImageTextToClassProblem, ImageTextTuple
from problems.image_text_to_class.clevr_dataset import CLEVRDataset
from misc.app_state import AppState
app_state = AppState()


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

        # instantiate CLEVRDataset class
        self.clevr_dataset = CLEVRDataset(self.set, self.clevr_dir, self.clevr_humans, self.embedding_type, self.random_embedding_dim)

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

    def collect_statistics(self, stat_col, data_tuple, logits, aux_tuple):
        """
        Collects accuracy.
        :param stat_col: Statistics collector.
        :param data_tuple: Data tuple containing inputs and targets.
        :param logits: Logits being output of the model.
        :param _: auxiliary tuple (aux_tuple) is not used in this function. 
        """
        stat_col['acc'] = self.calculate_accuracy(data_tuple, logits, aux_tuple)


        self.get_acc_per_family(data_tuple, aux_tuple, logits)

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

    def turn_on_cuda(self, data_tuple, aux_tuple):
        """
        Enables computations on GPU - copies the input and target matrices (from DataTuple) to GPU.

        :param data_tuple: Data tuple.
        :param aux_tuple: Auxiliary tuple (WARNING: Values stored in that variable will remain on CPU)
        :returns: Pair of Data and Auxiliary tuples (Data on GPU, Aux on CPU).
        """
        # Unpack tuples and copy data to GPU.
        inner_tuple, answers = data_tuple
        image_questions_tuple, questions_len = inner_tuple
        images, questions = image_questions_tuple

        gpu_images = images.cuda()
        gpu_questions = questions.cuda()
        gpu_answers = answers.cuda()

        gpu_image_text_tuple = ImageTextTuple(gpu_images, gpu_questions)
        gpu_inner_data_tuple = (gpu_image_text_tuple, questions_len)

        data_tuple = DataTuple(gpu_inner_data_tuple, gpu_answers)

        return data_tuple, aux_tuple

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

    params = {'batch_size': 64, 'CLEVR_dir': '/home/valbouy/CLEVR_v1.0', 'set': 'train', 'clevr_humans': False,
              'embedding_type': 'random', 'random_embedding_dim': 300}

    # create problem
    problem = CLEVR(params)

    # generate a batch
    data_tuple, aux_tuple = problem.generate_batch()
    inner_tuple, answers = data_tuple
    image_questions_tuple, questions_len = inner_tuple
    images, questions = image_questions_tuple

    print(questions.shape)
    #print(aux_tuple)

    # show a sample
    problem.show_sample(data_tuple, aux_tuple, sample_number=2)
    print('Unit test completed.')
