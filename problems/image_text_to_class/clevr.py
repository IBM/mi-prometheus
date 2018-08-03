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

"""clevr.py: This file contains 1 class:

          - Clevr (CLEVR Problem class): This class generates batches over a CLEVRDataset object.
            It also has a show_sample() method that displays a sample (image, question, answer).
  """
__author__ = "Vincent Albouy, Vincent Marois"


import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

# Add path to main project directory
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..', '..'))

from problems.problem import DataTuple
from problems.image_text_to_class.image_text_to_class_problem import ImageTextToClassProblem, ImageTextTuple
from problems.image_text_to_class.clevr_dataset import ClevrDataset
from misc.app_state import AppState
app_state = AppState()


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
        self.set = params['set']
        self.batch_size = params['batch_size']
        self.clevr_humans = params['clevr_humans']
        self.embedding_type = params['embedding_type']
        self.clevr_dataset = ClevrDataset(self.set, self.clevr_dir, self.clevr_humans, self.embedding_type)

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

    params = {'batch_size': 5, 'CLEVR_dir': 'CLEVR_v1.0', 'set': 'train', 'clevr_humans': False,
              'embedding_type': 'glove.6B.100d'}

    # create problem
    problem = Clevr(params)

    # generate a batch
    data_tuple, aux_tuple = problem.generate_batch()

    print(data_tuple)
    print(aux_tuple)

    problem.show_sample(data_tuple, aux_tuple, sample_number=2)
