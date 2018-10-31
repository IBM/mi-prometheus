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
stacked_attention_model.py: Implementation of a Stacked Attention Network (SAN). \

Inspiration drawn partially from the following paper:

@article{DBLP:journals/corr/YangHGDS15,
  author    = {Zichao Yang and
               Xiaodong He and
               Jianfeng Gao and
               Li Deng and
               Alexander J. Smola},
  title     = {Stacked Attention Networks for Image Question Answering},
  journal   = {CoRR},
  volume    = {abs/1511.02274},
  year      = {2015},
  url       = {http://arxiv.org/abs/1511.02274},
  archivePrefix = {arXiv},
  eprint    = {1511.02274},
  timestamp = {Mon, 13 Aug 2018 16:47:25 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/YangHGDS15},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

"""
__author__ = "Vincent Marois & Younes Bouhadjar"

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from miprometheus.models.model import Model
from miprometheus.models.vqa_baselines.stacked_attention_networks.stacked_attention_layer import StackedAttentionLayer


class StackedAttentionNetwork(Model):
    """
    Implementation of a Stacked Attention Networks (SAN).

    The three major components of SAN are:

        - the image model (CNN model, possibly pretrained),
        - the question model (LSTM based),
        - the stacked attention model.

    .. warning::

        This implementation has only been tested on ``SortOfCLEVR`` so far.

    """

    def __init__(self, params, problem_default_values_):
        """
        Constructor class of ``StackedAttentionNetwork`` model.

        - Parses the parameters,
        - Instantiates the CNN model: A simple, 4-layers one, or a pretrained one,
        - Instantiates an LSTM for the questions encoding,
        - Instantiates a 3-layers MLP as classifier.

        :param params: dict of parameters (read from configuration ``.yaml`` file).
        :type params: utils.ParamInterface

        :param problem_default_values_: default values coming from the ``Problem`` class.
        :type problem_default_values_: dict

        """
        # call base constructor
        super(StackedAttentionNetwork, self).__init__(params, problem_default_values_)

        # Parse default values received from problem.
        try:
            self.height = problem_default_values_['height']
            self.width = problem_default_values_['width']
            self.num_channels = problem_default_values_['num_channels']  # number of channels

            self.question_encoding_size = problem_default_values_['question_size']

            self.nb_classes = problem_default_values_['num_classes']

        except KeyError:
            self.logger.warning("Couldn't retrieve one or more value(s) from problem_default_values_.")

        self.name = 'StackedAttentionNetwork'

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

        # Retrieve attention layer parameters
        self.mid_features_attention = params['attention_layer']['nb_nodes']

        # Question encoding
        self.ffn = nn.Linear(in_features=output_question_dim, out_features=self.image_encoding_channels)

        # Instantiate class for attention
        self.apply_attention = StackedAttentionLayer(question_image_encoding_size=self.image_encoding_channels,
                                                     key_query_size=self.mid_features_attention)

        # Instantiate MLP for classifier
        input_size = self.image_encoding_channels

        self.fc1 = nn.Linear(in_features=input_size, out_features=params['classifier']['nb_hidden_nodes'])
        self.fc2 = nn.Linear(params['classifier']['nb_hidden_nodes'], params['classifier']['nb_hidden_nodes'])
        self.fc3 = nn.Linear(params['classifier']['nb_hidden_nodes'], self.nb_classes)

        self.data_definitions = {'images': {'size': [-1, self.num_channels, self.height, self.width], 'type': [torch.Tensor]},
                                 'questions': {'size': [-1, self.question_encoding_size], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1, self.nb_classes], 'type': [torch.Tensor]}
                                 }

    def forward(self, data_dict):
        """
        Runs the ``StackedAttentionNetwork`` model.

        :param data_dict: DataDict({'images', 'questions', **}) where:

            - images: [batch_size, num_channels, height, width],
            - questions: [batch_size, size_question_encoding]
        :type data_dict: utils.DataDict

        :returns: Predictions: [batch_size, output_classes]

        """

        images = data_dict['images'].type(self.app_state.dtype)
        questions = data_dict['questions']

        # 1. Encode the images
        encoded_images = self.cnn(images)

        # flatten the images
        encoded_images = encoded_images.view(encoded_images.size(0), encoded_images.size(1), -1).transpose(1, 2)

        # 2. Encode the questions

        # (h_0, c_0) are not provided -> default to zero
        encoded_question, _ = self.lstm(questions.unsqueeze(1))
        # take layer's last output
        encoded_question = encoded_question[:, -1, :]

        # 3. Go through the ``StackedAttentionLayer``.
        encoded_question = self.ffn(encoded_question)
        encoded_attention = self.apply_attention(encoded_images, encoded_question)

        # 4. Classify based on the result of the stacked attention layer
        x = F.relu(self.fc1(encoded_attention))
        x = F.relu(self.fc2(x))
        x = F.dropout(x)  # p=0.5
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
        plt.imshow(image.permute(1, 2, 0),
                   interpolation='nearest', aspect='auto')

        f = plt.figure()
        plt.title('Attention')

        width_height_attention = int(
            np.sqrt(self.apply_attention.visualize_attention.size(-2)))

        # get the attention of the 2 layers of stacked attention
        attention_visualize_layer1 = self.apply_attention.visualize_attention[sample, :, 0].detach(
        ).numpy()
        attention_visualize_layer2 = self.apply_attention.visualize_attention[sample, :, 1].detach(
        ).numpy()

        # reshape to get a 2D plot
        attention_visualize_layer1 = attention_visualize_layer1.reshape(
            width_height_attention, width_height_attention)
        attention_visualize_layer2 = attention_visualize_layer2.reshape(
            width_height_attention, width_height_attention)

        plt.title('1st attention layer')
        plt.imshow(attention_visualize_layer1,
                   interpolation='nearest', aspect='auto')

        f = plt.figure()

        plt.title('2nd attention layer')
        plt.imshow(attention_visualize_layer2,
                   interpolation='nearest', aspect='auto')

        # Plot!
        plt.show()


if __name__ == '__main__':
    """ Tests StackedAttentionNetwork on SortOfCLEVR"""

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
    from torch.utils.data.dataloader import DataLoader

    dataloader = DataLoader(dataset=sortofclevr, collate_fn=sortofclevr.collate_fn,
                            batch_size=batch_size, shuffle=True, num_workers=4)

    model_params = ParamInterface()
    model_params.add_config_params({'use_pretrained_cnn': False,

                                    'pretrained_cnn': {'name': 'resnet18', 'num_layers': 2},

                                    'lstm': {'hidden_size': 64, 'num_layers': 1, 'bidirectional': False,
                                             'dropout': 0},
                                    'attention_layer': {'nb_nodes': 128},
                                    'classifier': {'nb_hidden_nodes': 256}})

    # create model
    model = StackedAttentionNetwork(model_params, sortofclevr.default_values)

    for batch in dataloader:
        logits = model(batch)
        print(logits.shape)

        if model.plot(batch, logits):
            break
