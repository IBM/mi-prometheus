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

"""
multi_hops_stacked_attention_model.py: Implementation of a Stacked Attention Network (SAN).

This is a variant of the Stacked Attention Network model, in which several attention hops are done over the \
question words.

Inspiration drawn partially from the following paper:

@article{DBLP:journals/corr/KazemiE17,
  author    = {Vahid Kazemi and
               Ali Elqursh},
  title     = {Show, Ask, Attend, and Answer: {A} Strong Baseline For Visual Question
               Answering},
  journal   = {CoRR},
  volume    = {abs/1704.03162},
  year      = {2017},
  url       = {http://arxiv.org/abs/1704.03162},
  archivePrefix = {arXiv},
  eprint    = {1704.03162},
  timestamp = {Mon, 13 Aug 2018 16:47:10 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/KazemiE17},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

"""
__author__ = "Vincent Marois & Younes Bouhadjar"

import torch
import numpy as np
import torch.nn as nn

from miprometheus.models.model import Model
from miprometheus.models.vqa_baselines.stacked_attention_networks.stacked_attention_layer import StackedAttentionLayer


class MultiHopsStackedAttentionNetwork(Model):
    """
    Implementation of a Stacked Attention Networks (SAN), with several attention hops over the \
    question words.

    The implementation details are very similar to the `StackedAttentionNetwork``, to the difference that \
    it uses an LSTMCell instead of an LSTM.

    .. warning::

        This implementation has only been tested on ``ShapeColorQuery`` so far.

    """

    def __init__(self, params, problem_default_values_):
        """
        Constructor class of ``MultiHopsStackedAttentionNetwork`` model.

        - Parses the parameters,
        - Instantiates the CNN model: A simple, 4-layers one, or a pretrained one,
        - Instantiates an LSTMCell for the questions encoding,
        - Instantiates a 3-layers MLP as classifier.

        :param params: dict of parameters (read from configuration ``.yaml`` file).
        :type params: utils.ParamInterface

        :param problem_default_values_: default values coming from the ``Problem`` class.
        :type problem_default_values_: dict

        """
        # call base constructor
        super(MultiHopsStackedAttentionNetwork, self).__init__(params, problem_default_values_)

        # Parse default values received from problem.
        try:
            self.height = problem_default_values_['height']
            self.width = problem_default_values_['width']
            self.num_channels = problem_default_values_['num_channels']  # number of channels

            self.question_encoding_size = problem_default_values_['question_size']

            self.nb_classes = problem_default_values_['num_classes']

            self.num_words = problem_default_values_['seq_length']

        except KeyError:
            self.num_words = params['default_nb_hops']
            self.logger.warning("Couldn't retrieve one or more value(s) from problem_default_values_.")

        self.name = 'MultiHopsStackedAttentionNetwork'

        # Instantiate CNN for image encoding
        if params['use_pretrained_cnn']:
            from miprometheus.models.vqa_baselines import PretrainedImageEncoding
            self.cnn = PretrainedImageEncoding(params['pretrained_cnn']['name'], params['pretrained_cnn']['num_layers'])
            self.image_encoding_channels = self.cnn.get_output_nb_filters()

        else:
            from miprometheus.models import ConvInputModel
            self.cnn = ConvInputModel()
            self.image_encoding_channels = self.cnn.get_output_nb_filters()

        # Instantiate LSTM for question encoding
        self.hidden_size = self.image_encoding_channels

        self.lstm = nn.LSTMCell(input_size=self.question_encoding_size,
                                hidden_size=self.hidden_size,
                                bias=True)

        # Retrieve attention layer parameters
        self.mid_features_attention = params['attention_layer']['nb_nodes']

        # Instantiate class for attention
        self.apply_attention = StackedAttentionLayer(question_image_encoding_size=self.image_encoding_channels,
                                                     key_query_size=self.mid_features_attention)

        # Instantiate MLP for classifier
        input_size = (self.num_words + 1) *self.image_encoding_channels

        self.fc1 = nn.Linear(in_features=input_size, out_features=params['classifier']['nb_hidden_nodes'])
        self.fc2 = nn.Linear(params['classifier']['nb_hidden_nodes'], params['classifier']['nb_hidden_nodes'])
        self.fc3 = nn.Linear(params['classifier']['nb_hidden_nodes'], self.nb_classes)

        self.data_definitions = {
            'images': {'size': [-1, self.num_channels, self.height, self.width], 'type': [torch.Tensor]},
            'questions': {'size': [-1, 3, self.question_encoding_size], 'type': [torch.Tensor]},
            'targets': {'size': [-1, self.nb_classes], 'type': [torch.Tensor]}
            }

    def init_hidden_states(self, batch_size):
        """
        Initialize the hidden and cell states of the LSTM to 0.

        :param batch_size: Size of the batch.
        :type batch_size: int

        :return: hx, cx: hidden and cell states initialized to 0.

        """
        hx = torch.zeros(batch_size, self.hidden_size).type(self.app_state.dtype)
        cx = torch.zeros(batch_size, self.hidden_size).type(self.app_state.dtype)

        return hx, cx

    def forward(self, data_dict):
        """
        Runs the ``MultiHopsStackedAttentionNetwork`` model.

        :param data_dict: DataDict({'images', 'questions', **}) where:

            - images: [batch_size, num_channels, height, width],
            - questions: [batch_size, size_question_encoding]
        :type data_dict: utils.DataDict

        :returns: Predictions: [batch_size, output_classes]

        """

        images = data_dict['images'].type(self.app_state.dtype)
        questions = data_dict['questions']

        # get batch size
        batch_size = images.shape[0]

        # 1. Encode the images
        encoded_images = self.cnn(images)
        # flatten the images
        encoded_images = encoded_images.view(encoded_images.size(0), encoded_images.size(1), -1).transpose(1, 2)

        # 2. Encode the questions
        v_features = None

        # initialize the LSTM states
        hx, cx = self.init_hidden_states(batch_size)

        for i in range(questions.size(1)):

            hx, cx = self.lstm(questions[:, i, :], (hx, cx))
            # 3. Go through the ``StackedAttentionLayer``.
            v = self.apply_attention(encoded_images, hx.squeeze(1))

            if v_features is None:
                v_features = v
            else:
                v_features = torch.cat((v_features, v), dim=-1)

                # 4. Classify based on the result of the stacked attention layer
        combined = torch.cat([v_features, hx], dim=1)
        x = torch.nn.functional.relu(self.fc1(combined))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.dropout(x)  # p=0.5
        logits = self.fc3(x)

        return logits

    def plot(self, data_dict, predictions, sample=0):
        """
        Displays the image, the predicted & ground truth answers.

        :param data_dict: DataDict({'images', 'questions', 'targets'}) where:

            - images: [batch_size, num_channels, height, width],
            - questions: [batch_size, size_question_encoding]
            - targets: [batch_size]

        :type data_dict: utils.DataDict

        :param predictions: Prediction.
        :type predictions: torch.tensor

        :param sample: Index of sample in batch (DEFAULT: 0).
        :type sample:int
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
        print(type(image))
        plt.imshow(image.permute(1, 2, 0),
                   interpolation='nearest', aspect='auto')

        # Plot!
        plt.show()


if __name__ == '__main__':
    """ Tests MultiHopsStackedAttentionNetwork on ShapeColorQuery"""

    # "Loaded parameters".
    from miprometheus.utils.param_interface import ParamInterface
    from miprometheus.utils.app_state import AppState
    app_state = AppState()
    app_state.visualize = False
    from miprometheus.problems import ShapeColorQuery
    problem_params = ParamInterface()
    problem_params.add_config_params({'data_folder': '~/data/shape-color-query/',
                                      'split': 'train',
                                      'regenerate': False,
                                      'dataset_size': 10000,
                                      'img_size': 128})

    # create problem
    shapecolorquery = ShapeColorQuery(problem_params)

    batch_size = 64

    # wrap DataLoader on top of this Dataset subclass
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset=shapecolorquery, collate_fn=shapecolorquery.collate_fn,
                            batch_size=batch_size, shuffle=True, num_workers=4)

    model_params = ParamInterface()
    model_params.add_config_params({'use_pretrained_cnn': False,

                                    'pretrained_cnn': {'name': 'resnet18', 'num_layers': 2},

                                    'attention_layer': {'nb_nodes': 128},
                                    'classifier': {'nb_hidden_nodes': 256},
                                    'default_nb_hops': 3})

    # create model
    model = MultiHopsStackedAttentionNetwork(model_params, shapecolorquery.default_values)

    for batch in dataloader:
        logits = model(batch)
        print(logits.shape)

        if model.plot(batch, logits):
            break
