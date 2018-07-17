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

"""stacked_attention_vqa.py: implements stacked attention model https://arxiv.org/abs/1511.02274 """
__author__ = "Younes Bouhadjar"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from misc.param_interface import ParamInterface


from models.stacked_attention_vqa.image_encoding import ImageEncoding
from models.stacked_attention_vqa.attention import StackedAttention, Attention
from models.model import Model
from misc.app_state import AppState


class StackedAttentionVQA(Model):

    def __init__(self, params):
        """ Implementation of simple vqa model with attention, it performs the following steps:

           step1: image encoding
           step2: question encoding if needed
           step3: apply attention, the question is used as a query and image as key
           step4: classifier, create the probabilities

        """

        super(StackedAttentionVQA, self).__init__(params)

        # Retrieve attention and image/questions parameters
        self.encoded_question_size = 13
        self.num_channels_image = 3
        self.mid_features = 64
        self.image_encoding_channels = 256

        # LSTM parameters (if use_question_encoding is True)
        self.hidden_size = self.encoded_question_size
        self.word_embedded_size = 7
        self.num_layers = 3
        self.use_question_encoding = params['use_question_encoding']

        # Instantiate class for image encoding
        self.image_encoding = ImageEncoding()

        # Instantiate class for question encoding
        self.lstm = nn.LSTM(self.word_embedded_size, self.hidden_size, self.num_layers, batch_first=True)

        # Question encoding
        self.ffn = nn.Linear(self.encoded_question_size, self.image_encoding_channels)

        # Instantiate class for attention
        self.apply_attention = StackedAttention(
            question_image_encoding_size=self.image_encoding_channels,
            key_query_size=self.mid_features
        )

        # Instantiate classifier class
        self.classifier = Classifier(
            in_features=256, #+ self.encoded_question_size,
            mid_features=256,
            out_features=10)

        self.encoded_image_attention_visualize = []

    def forward(self, data_tuple):
        """
        Runs the stacked_attention model and plots if necessary

        :param data_tuple: Tuple containing images [batch_size, num_channels, height, width] and questions [batch_size, size_question_encoding]
        :returns: output [batch_size, output_classes]
        """

        (images, questions), _ = data_tuple

        # step1 : encode image
        encoded_images = self.image_encoding(images)

        # step2 : encode question
        if self.use_question_encoding:
            batch_size = images.size(0)
            hx, cx = self.init_hidden_states(batch_size)
            encoded_question, _ = self.lstm(questions, (hx, cx))
            encoded_question = encoded_question[:, -1, :]
        else:
            encoded_question = questions

        # step3 : apply attention
        encoded_question = self.ffn(encoded_question)
        encoded_attention = self.apply_attention(encoded_images, encoded_question)
        if self.app_state.visualize:
            self.encoded_image_attention_visualize = encoded_attention

        # step 4: classifying based in the encoded questions and attention
        answer = self.classifier(encoded_attention)

        return answer

    def init_hidden_states(self, batch_size):
        """
        Initialize hidden state ans cell state of the stacked LSTM used for question encoding

        :param batch_size: Size of the batch in given iteraction/epoch.
        :return: hx, cx: hidden state and cell state of a stacked LSTM [num_layers, batch_size, hidden_size]
        """

        dtype = AppState().dtype
        hx = torch.randn(self.num_layers, batch_size, self.hidden_size).type(dtype)
        cx = torch.randn(self.num_layers, batch_size, self.hidden_size).type(dtype)

        return hx, cx

    def plot(self, data_tuple, predictions, sample_number=0):
        """
        Simple plot - shows MNIST image with target and actual predicted class.

        :param data_tuple: Data tuple containing input and target batches.
        :param predictions: Prediction.
        :param sample_number: Number of sample in batch (DEFAULT: 0)
        """
        # Check if we are supposed to visualize at all.
        if not self.app_state.visualize:
            return False
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # Unpack tuples.
        (images, texts), targets = data_tuple

        # Get sample.
        image = images[sample_number].cpu().detach().numpy()
        target = targets[sample_number].cpu().detach().numpy()
        prediction = predictions[sample_number].cpu().detach().numpy()

        # Show data.
        plt.title('Prediction: {} (Target: {})'.format(np.argmax(prediction), target))
        plt.imshow(image.transpose(1,2,0), interpolation='nearest', aspect='auto')

        f = plt.figure()
        plt.title('Attention')

        attention_visualize = self.encoded_image_attention_visualize[0].view(16,16).detach().numpy()
        plt.imshow(attention_visualize, interpolation='nearest', aspect='auto')

        # Plot!
        plt.show()
        exit()

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features):
        """

        Predicts the final answer to the question, based on the question and the attention.

        :param in_features: input size of the first feed forward layer
        :param mid_features: input size of the intermediates feed forward layers
        :param out_features: output size
        """

        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(in_features, mid_features)
        self.fc2 = nn.Linear(mid_features, mid_features)
        self.fc3 = nn.Linear(mid_features, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)


if __name__ == '__main__':
    # Set visualization.
    AppState().visualize = True

    # Test base model.
    params = ParamInterface()
    params.add_custom_params({'use_question_encoding': False})

    # model
    model = StackedAttentionVQA(params)

    while True:
        # Generate new sequence.
        # "Image" - batch x channels x width x height
        input_np = np.random.binomial(1, 0.5, (2, 3, 128,  128))
        image = torch.from_numpy(input_np).type(torch.FloatTensor)

        # Question
        if params['use_question_encoding']:
            questions_np = np.random.binomial(1, 0.5, (2, 13, 7))
        else:
            questions_np = np.random.binomial(1, 0.5, (2, 13))

        questions = torch.from_numpy(questions_np).type(torch.FloatTensor)

        # Target.
        target = torch.randint(10, (10,), dtype=torch.int64)

        dt = (image, questions), target
        # prediction.
        prediction = model(dt)

        # Plot it and check whether window was closed or not.
        if model.plot(dt, prediction):
            break