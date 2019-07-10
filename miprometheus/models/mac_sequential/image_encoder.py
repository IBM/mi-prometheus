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
input_unit.py: Implementation of the input unit for the MAC network. Cf https://arxiv.org/abs/1803.03067 for \
the reference paper.
"""
__author__ = "Vincent Marois"
import torch
from torch.nn import Module
import torch.nn as nn
import numpy as np

from miprometheus.models.mac_sequential.utils_mac import linear


class ImageEncoder(Module):
    """
    Implementation of the ``InputUnit`` of the MAC network.
    """

    def __init__(self, dim, embedded_dim):
        """
        Constructor for the ``InputUnit``.

        :param dim: global 'd' hidden dimension
        :type dim: int

        :param embedded_dim: dimension of the word embeddings.
        :type embedded_dim: int

        """

        # call base constructor
        super(ImageEncoder, self).__init__()

        self.dim = dim

        # Number of channels in input Image
        self.image_channels = 3

        # CNN number of channels
        self.visual_processing_channels = [32, 64, 64, 128]

        # LSTM hidden units.
        self.lstm_hidden_units = 64

        # Input Image size
        self.image_size = [112, 112]

        # history states
        self.cell_states = []

        # Visual memory shape. height x width.
        self.vstm_shape = np.array(self.image_size)
        for channel in self.visual_processing_channels:
            self.vstm_shape = np.floor((self.vstm_shape) / 2)
        self.vstm_shape = [int(dim) for dim in self.vstm_shape]

        # Number of GRU units in controller
        self.controller_output_size = 768

        self.VisualProcessing(self.image_channels,
                              self.visual_processing_channels,
                              self.lstm_hidden_units * 2,
                              self.controller_output_size * 2,
                              self.vstm_shape)

        # Initialize weights and biases
        # -----------------------------------------------------------------
        # Visual processing
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain('relu'))

        self.conv1.bias.data.fill_(0.01)
        self.conv2.bias.data.fill_(0.01)
        self.conv3.bias.data.fill_(0.01)
        self.conv4.bias.data.fill_(0.01)

    def forward(self, images):
        """
        Forward pass of the ``InputUnit``.

        :param questions: tensor of the questions words, shape [batch_size x maxQuestionLength x embedded_dim].
        :type questions: torch.tensor

        :param questions_len: Unpadded questions length.
        :type questions_len: list

        :param feature_maps: [batch_size x nb_kernels x feat_H x feat_W] coming from `ResNet101`.
        :type feature_maps: torch.tensor

        :return:

            - question encodings: [batch_size x 2*dim] (torch.tensor),
            - word encodings: [batch_size x maxQuestionLength x dim] (torch.tensor),
            - images_encodings: [batch_size x nb_kernels x (H*W)] (torch.tensor).


        """
        batch_size = images.shape[0]

        # Cog like CNN - cf  cog  model
        x = self.conv1(images)
        x = self.maxpool1(x)
        x = nn.functional.relu(self.batchnorm1(x))
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = nn.functional.relu(self.batchnorm2(x))
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = nn.functional.relu(self.batchnorm3(x))
        x = self.conv4(x)
        # out_conv4 = self.conv4(out_batchnorm3)
        feature_maps = self.maxpool4(x)


        # reshape feature maps as channels first
        feature_maps = feature_maps.view(batch_size, self.dim, -1)

        # return everything
        return feature_maps

    def VisualProcessing(self, in_channels, layer_channels, feature_control_len, spatial_control_len, output_shape):
        """
        Defines all layers pertaining to visual processing.

        :param in_channels: Number of channels in images in dataset. Usually 3 (RGB).
        :type in_channels: Int

        :param layer_channels: Number of feature maps in the CNN for each layer.
        :type layer_channels: List of Ints

        :param feature_control_len: Input size to the Feature Attention linear layer.
        :type feature_control_len: Int

        :param spatial_control_len: Input size to the Spatial Attention linear layer.
        :type spatial_control_len: Int

        :param output_shape: Output dimensions of feature maps of last layer.
        :type output_shape: Tuple of Ints

        """
        # Initial Norm
        # self.batchnorm0 = nn.BatchNorm2d(3)

        # First Layer
        self.conv1 = nn.Conv2d(in_channels, layer_channels[0], 3,
                               stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.maxpool1 = nn.MaxPool2d(2,
                                     stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.batchnorm1 = nn.BatchNorm2d(layer_channels[0])

        # Second Layer
        self.conv2 = nn.Conv2d(layer_channels[0], layer_channels[1], 3,
                               stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.maxpool2 = nn.MaxPool2d(2,
                                     stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.batchnorm2 = nn.BatchNorm2d(layer_channels[1])

        # Third Layer
        self.conv3 = nn.Conv2d(layer_channels[1], layer_channels[2], 3,
                               stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.maxpool3 = nn.MaxPool2d(2,
                                     stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.batchnorm3 = nn.BatchNorm2d(layer_channels[2])

        # Fourth Layer
        self.conv4 = nn.Conv2d(layer_channels[2], layer_channels[3], 3,
                               stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.maxpool4 = nn.MaxPool2d(2,
                                     stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.batchnorm4 = nn.BatchNorm2d(layer_channels[3])

