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
Image_encoder.py: Implementation of the ImageEncoder for the VWM network.

"""

__author__ = "Vincent Albouy"
import torch
from torch.nn import Module
import torch.nn as nn

class ImageEncoder(Module):

    """
    Implementation of the ``ImageEncoder`` of the VWM network.
    """

    def __init__(self, dim):
        """
        Constructor for the ``ImageEncoder``.

        :param dim: global 'd' hidden dimension
        :type dim: int

        """

        # call base constructor
        super(ImageEncoder, self).__init__()

        self.dim = dim

        # Number of channels in input Image RGB
        self.image_channels = 3 # [R,G,B]

        # CNN number of channels - Parameters for the CNN
        self.visual_processing_channels = [32, 64, 64, 128]

        #call utility class to buil the CNN layers
        self.VisualProcessing(self.image_channels,
                              self.visual_processing_channels)


        # Initialize weights and biases
        # -----------------------------------------------------------------
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
        Forward pass of the ``ImageEncoder``.

        :param questions: tensor of the questions words, shape [batch_size x maxQuestionLength x embedded_dim].
        :type questions: torch.tensor
              
        :return:
        :param feature_maps: [batch_size x nb_kernels x feat_H x feat_W].
        :type feature_maps: torch.tensor

    
        """
        batch_size = images.shape[0]

        # Cog like CNN - cf cog  model
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
        feature_maps = self.maxpool4(x)


        # reshape feature maps as channels first
        feature_maps = feature_maps.view(batch_size, self.dim, -1)

        # return feature_maps
        return feature_maps

    def VisualProcessing(self, in_channels, layer_channels):
        """
        Defines all layers pertaining to visual processing.

        :param in_channels: Number of channels in images in dataset. Usually 3 (RGB).
        :type in_channels: Int

        :param layer_channels: Number of feature maps in the CNN for each layer.
        :type layer_channels: List of Ints

        """

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

