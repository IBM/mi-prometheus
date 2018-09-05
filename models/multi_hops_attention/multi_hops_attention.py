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

"""multi_hops_attention.py: implement multi hops stacked attention model"""
__author__ = "Younes Bouhadjar"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from misc.param_interface import ParamInterface

from models.multi_hops_attention.image_encoding import ImageEncoding
from models.multi_hops_attention.attention import StackedAttention
from models.model import Model
from misc.app_state import AppState


class MultiHopsAttention(Model):
    """ Implementation of simple vqa model with multi attention hops over the words of the question, it performs the following steps: \n
       step1: image encoding \n
       step2: word encoding \n
       step3: apply attention, an attention over the image is generates for every word, after that all the attentions are concatenated \n
       step4: classifier, create the probabilities
    """

    def __init__(self, params):
        super(MultiHopsAttention, self).__init__(params)

        """
        Constructor of MultiHopsAttention class
        :param params dictionary of inputs
        """

        # Retrieve attention and image/questions parameters
        self.image_encoding_channels = 256
        self.encoded_question_size = self.image_encoding_channels
        self.num_channels_image = 3
        self.mid_features = 512
        self.num_words = 3

        # LSTM parameters
        self.hidden_size = self.image_encoding_channels
        self.word_embedded_size = 7
        self.num_layers = 3

        # Instantiate class for image encoding
        self.image_encoding = ImageEncoding()

        # Instantiate class for question encoding
        self.lstm = nn.LSTMCell(self.word_embedded_size, self.hidden_size)

        # Instantiate class for attention
        self.apply_attention = StackedAttention(
            question_image_encoding_size=self.image_encoding_channels,
            key_query_size=self.mid_features
        )

        # Instantiate class for classifier
        self.classifier = Classifier(
            in_features=self.num_words*(self.image_encoding_channels) + self.image_encoding_channels,
            mid_features=256,
            out_features=10)

    def forward(self, data_tuple):
        """
        Runs the multi hops attention model and plots if necessary

        :param data_tuple: Tuple containing images [batch_size, num_channels, height, width] and questions [batch_size, size_question_encoding]
        :returns: output [batch_size, output_classes]
        """

        (images, questions), _ = data_tuple

        # step1 : encode image
        encoded_images = self.image_encoding(images)

        # Initial hidden_state for question encoding
        batch_size = images.size(0)
        hx, cx = self.init_hidden_states(batch_size)

        # step2 : encode question
        v_features = None
        for i in range(questions.size(1)):
            # step 2: encode words
            hx, cx = self.lstm(questions[:, i, :], (hx, cx))

            v = self.apply_attention(encoded_images, hx)

            if v_features is None:
                 v_features = v
            else:
                v_features = torch.cat((v_features, v), dim=-1)

        # step 4: classifying based in the encoded questions and attention
        combined = torch.cat([v_features, hx], dim=1)
        answer = self.classifier(combined)

        return answer

    def init_hidden_states(self, batch_size):
        """
        Initialize hidden state ans cell state of the stacked LSTM used for question encoding

        :param batch_size: Size of the batch in given iteraction/epoch.
        :return: hx, cx: hidden state and cell state of a stacked LSTM [num_layers, batch_size, hidden_size]
        """

        dtype = AppState().dtype
        hx = torch.randn(batch_size, self.hidden_size).type(dtype)
        cx = torch.randn(batch_size, self.hidden_size).type(dtype)

        return hx, cx

    def plot(self, data_tuple, predictions, sample_number=0):
        """
        Simple plot - shows SortOfClevr image, question, answer and prediction

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
        (images, questions), targets = data_tuple

        # Get sample.
        image = images[sample_number]
        target = targets[sample_number]
        prediction = predictions[sample_number]
        question = questions[sample_number]

        # Show data.
        plt.title('Prediction: {} (Target: {})'.format(prediction, target))
        plt.xlabel('Q: {} )'.format(question))
        plt.imshow(image.transpose(1,2,0), interpolation='nearest', aspect='auto')

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
        """
        Apply a set of feed forward layers to the combined question/attention to obtain probabilities over the classes output

        :param x: a combination of the attention and question
        :return: Prediction of the answer [batch_size, num_classes]
        """

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

    # model
    model = MultiHopsAttention(params)

    while True:
        # Generate new sequence.
        # "Image" - batch x channels x width x height
        input_np = np.random.binomial(1, 0.5, (1, 3, 128,  128))
        image = torch.from_numpy(input_np).type(torch.FloatTensor)

        #Question
        questions_np = np.random.binomial(1, 0.5, (1, 3, 7))
        questions = torch.from_numpy(questions_np).type(torch.FloatTensor)

        # Target.
        target = torch.randint(10, (10,), dtype=torch.int64)

        dt = (image, questions), target
        # prediction.
        prediction = model(dt)

        # Plot it and check whether window was closed or not.
        if model.plot(dt, prediction):
            break