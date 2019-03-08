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

"""cnn_lstm.py: A simple bag-of-words baseline for Visual Question Answering (VQA). \
This baseline concatenates the word features from the question and CNN features \
from the image to predict the answer.

Inspiration drawn partially from the following paper:

 @article{DBLP:journals/corr/ZhouTSSF15,
  author    = {Bolei Zhou and
               Yuandong Tian and
               Sainbayar Sukhbaatar and
               Arthur Szlam and
               Rob Fergus},
  title     = {Simple Baseline for Visual Question Answering},
  journal   = {CoRR},
  volume    = {abs/1512.02167},
  year      = {2015},
  url       = {http://arxiv.org/abs/1512.02167},
  archivePrefix = {arXiv},
  eprint    = {1512.02167},
  timestamp = {Mon, 13 Aug 2018 16:47:29 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/ZhouTSSF15},
  bibsource = {dblp computer science bibliography, https://dblp.org}}

"""
__author__ = "Vincent Marois & Younes Bouhadjar"

import torch
import numpy as np
import torch.nn as nn

from miprometheus.models.model import Model
from miprometheus.models.relational_net.conv_input_model import ConvInputModel


class CNN_LSTM(Model):
    """
    Implementation of a simple VQA baseline, globally following these steps:

        1. Image Encoding, using a CNN model,
        2. Question Encoding (if specified) using a LSTM,
        3. Concatenates the two features vectors and pass then through a MLP to produce the predictions.

    .. warning::

        The CNN model used in this implementation is the one from the Relational Network model \
        (implementation in models.relational_net.conv_input_model.py), constituted of 4 convolutional \
        layers (with batch normalization).

        Altough the cited paper above mentions GoogLeNet & VGG as other CNN models, they are not supported \
        for now. It is planned in a future release to add support for ``torchvision`` models.

        This implementation has only been tested on ``SortOfCLEVR`` for now.


    """

    def __init__(self, params, problem_default_values_={}):
        """
        Constructor of the ``CNN_LSTM`` model.

        Parses the parameters, instantiates the LSTM & CNN model, alongside with the MLP classifier.


        :param params: dict of parameters (read from configuration ``.yaml`` file).
        :type params: utils.ParamInterface

        :param problem_default_values_: default values coming from the ``Problem`` class.
        :type problem_default_values_: dict

        """
        # call base constructor
        super(CNN_LSTM, self).__init__(params, problem_default_values_)
        self.name = 'CNN_LSTM'


        # Parse default values received from problem.
        try:
            self.height = problem_default_values_['height']
            self.width = problem_default_values_['width']
            self.num_channels = problem_default_values_['depth']  # number of channels

            self.question_encoding_size = problem_default_values_['question_encoding_size']

            self.nb_classes = problem_default_values_['num_classes']

        except Exception as ex:
            self.logger.error("Couldn't retrieve '{}' from 'problem default values':".format(ex))
            exit(1)

        # Instantiate CNN for image encoding
        self.cnn = ConvInputModel()
        output_height, output_width = self.cnn.get_output_shape(self.height, self.width)

        feature_maps_flattened_dim = self.cnn.get_output_nb_filters() * output_height * output_width

        # whether to use question encoding or not.
        self.use_question_encoding = params['use_question_encoding']

        if self.use_question_encoding:

            # Instantiate LSTM for question encoding
            self.hidden_size = params['lstm']['hidden_size']
            self.num_layers = params['lstm']['num_layers']

            self.bidirectional = params['lstm']['bidirectional']
            if self.bidirectional:
                self.num_dir = 2
            else:
                self.num_dir = 1

            self.lstm = nn.LSTM(input_size=self.question_encoding_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                bias=True,
                                batch_first=True,
                                dropout=params['lstm']['dropout'],
                                bidirectional=self.bidirectional)

            output_question_dim = self.num_dir * self.hidden_size

        else:
            output_question_dim = self.question_encoding_size

        # Instantiate MLP for classifier
        input_size = feature_maps_flattened_dim + output_question_dim

        self.fc1 = nn.Linear(in_features=input_size, out_features=params['classifier']['nb_hidden_nodes'])
        self.fc2 = nn.Linear(params['classifier']['nb_hidden_nodes'], params['classifier']['nb_hidden_nodes'])
        self.fc3 = nn.Linear(params['classifier']['nb_hidden_nodes'], self.nb_classes)

        self.data_definitions = {'images': {'size': [-1, self.num_channels, self.height, self.width], 'type': [torch.Tensor]},
                                 'questions': {'size': [-1, self.question_encoding_size], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1, self.nb_classes], 'type': [torch.Tensor]}
                                 }

    def forward(self, data_dict):
        """
        Runs the ``CNN_LSTM`` model.

        :param data_dict: DataDict({'images', 'questions', ...}) where:

            - images: [batch_size, num_channels, height, width],
            - questions: [batch_size, size_question_encoding]
        :type data_dict: utils.DataDict

        :returns: Predictions: [batch_size, output_classes]

        """
        images = data_dict['images'].type(self.app_state.dtype)
        questions = data_dict['questions']

        print(images.size())
        print(questions.size())


        # get batch_size
        batch_size = images.size(0)

        # 1. Encode the images
        encoded_images = self.cnn(images)
        # flatten images
        encoded_image_flattened = encoded_images.view(batch_size, -1)

        # 2. Encode the questions
        if self.use_question_encoding:

            # (h_0, c_0) are not provided -> default to zero
            encoded_question, _ = self.lstm(questions)
            # take layer's last output
            encoded_question = encoded_question[:, -1, :]
        else:
            encoded_question = questions

        # 3. Classify based on the encodings
        combined = torch.cat([encoded_image_flattened, encoded_question], dim=-1)

        x = torch.nn.functional.relu(self.fc1(combined))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.dropout(x)  # p=0.5
        logits = self.fc3(x)

        return logits

    def plot(self, data_dict, predictions, sample=0):
        """
        Displays the image, the predicted & ground truth answers.

        :param data_dict: DataDict({'images', 'questions', 'targets'}) where:

            - images: [batch_size, num_channels, height, width]
            - questions: [batch_size, size_question_encoding]
            - targets: [batch_size]

        :type data_dict: utils.DataDict

        :param predictions: Prediction.
        :type predictions: torch.tensor

        :param sample: Index of sample in batch (DEFAULT: 0).
        :type sample: int

        """
        # Check if we are supposed to visualize at all.
        if not self.app_state.visualize:
            return False
        import matplotlib.pyplot as plt

        images = data_dict['images']
        questions = data_dict['questions']
        targets = data_dict['targets']

        # Get sample.
        image = images[sample]
        target = targets[sample]
        prediction = np.argmax(predictions[sample].detach().numpy())
        question = questions[sample]

        # Show data.
        plt.title('Prediction: {} (Target: {})'.format(prediction, target))
        plt.xlabel('Q: {} )'.format(question))
        plt.imshow(image.permute(1, 2, 0),
                   interpolation='nearest', aspect='auto')

        # Plot!
        plt.show()
        exit()


if __name__ == '__main__':
    """ Tests CNN_LSTM on SortOfCLEVR"""

    # "Loaded parameters".
    from miprometheus.utils.param_interface import ParamInterface
    from miprometheus.utils.app_state import AppState
    app_state = AppState()
    app_state.visualize = True
    from miprometheus.problems.image_text_to_class.sort_of_clevr import SortOfCLEVR
    problem_params = ParamInterface()
    problem_params.add_config_params({'data_folder': '~/data/sort-of-clevr/',
                                      'split': 'train',
                                      'regenerate': False,
                                      'dataset_size': 10000,
                                      'img_size': 128})

    # create problem
    sortofclevr = SortOfCLEVR(problem_params)

    batch_size = 64

    # wrap DataLoader on top of this Dataset subclass
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset=sortofclevr, collate_fn=sortofclevr.collate_fn,
                            batch_size=batch_size, shuffle=True, num_workers=4)

    model_params = ParamInterface()
    model_params.add_config_params({'use_question_encoding': True,
                                    'lstm': {'hidden_size': 64, 'num_layers': 1, 'bidirectional': False,
                                             'dropout': 0},
                                    'classifier': {'nb_hidden_nodes': 256}})

    # create model
    model = CNN_LSTM(model_params, sortofclevr.default_values)

    for batch in dataloader:
        logits = model(batch)
        print(logits.shape)

        if model.plot(batch, logits):
            break
