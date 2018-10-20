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

"""conv_input_model.py: contains CNN model for the ``RelationalNetwork``."""
__author__ = "Vincent Marois"

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mip.utils.app_state import AppState


class ConvInputModel(nn.Module):
    """
    Simple 4 layers CNN for image encoding in the ``RelationalNetwork`` model.

    """

    def __init__(self):
        """
        Constructor.

        Defines the 4 convolutional layers and batch normalization layers.

        This implementation is inspired from the description in the section \
        'Supplementary Material - CLEVR from pixels' in the reference paper \
        (https://arxiv.org/pdf/1706.01427.pdf).


        """

        # call base constructor
        super(ConvInputModel, self).__init__()

        # Note: formula for computing the output size is O = floor((W - K + 2P)/S + 1)
        # W is the input height/length, K is the filter size, P is the padding, and S is the stride

        # define layers
        # input image size is indicated as 128 x 128 in the paper for this
        # model
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1)
        # output shape should be [24 x 64 x 64]
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        # output shape should be [24 x 32 x 32]
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        # output shape should be [24 x 16 x 16]
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        # output shape should be [24 x 8 x 8]
        self.batchNorm4 = nn.BatchNorm2d(24)

    def get_output_nb_filters(self):
        """
        :return: The number of filters of the last conv layer.
        """
        return self.conv4.out_channels

    def get_output_shape(self, height, width):
        """
        Getter method which computes the output height & width of the features maps.

        :param height: Input image height.
        :type height: int

        :param width: Input image width.
        :type width: int

        :return: height, width of the produced feature maps.

        """
        def get_output_dim(dim, kernel_size, stride, padding):
            """
            Using the convolution formula to compute the output dim with the specified kernel_size, stride, padding.

            Assuming dilatation=1.
            """
            return np.floor(((dim + 2*padding - kernel_size)/stride) + 1)

        height1 = get_output_dim(height, self.conv1.kernel_size[0], self.conv1.stride[0], self.conv1.padding[0])
        width1 = get_output_dim(width, self.conv1.kernel_size[1], self.conv1.stride[1], self.conv1.padding[1])

        height2 = get_output_dim(height1, self.conv2.kernel_size[0], self.conv2.stride[0], self.conv2.padding[0])
        width2 = get_output_dim(width1, self.conv2.kernel_size[1], self.conv2.stride[1], self.conv2.padding[1])

        height3 = get_output_dim(height2, self.conv3.kernel_size[0], self.conv3.stride[0], self.conv3.padding[0])
        width3 = get_output_dim(width2, self.conv3.kernel_size[1], self.conv3.stride[1], self.conv3.padding[1])

        height4 = get_output_dim(height3, self.conv4.kernel_size[0], self.conv4.stride[0], self.conv4.padding[0])
        width4 = get_output_dim(width3, self.conv4.kernel_size[1], self.conv4.stride[1], self.conv4.padding[1])

        return height4, width4

    def forward(self, img):
        """
        Forward pass of the CNN.
        :param img: images to pass through the CNN layers. Should be of size [N, 3, 128, 128].
        :type img: torch.tensor

        :return: output of the CNN. Should be of size [N, 24, 8, 8].
        """
        x = self.conv1(img)
        x = self.batchNorm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = F.relu(x)

        return x


if __name__ == '__main__':
    """
    Unit Test for the ``ConvInputModel``.
    """

    # "Image" - batch x channels x width x height
    batch_size = 64
    img_size = 128

    input_np = np.random.binomial(1, 0.5, (batch_size, 3, img_size, img_size))
    image = torch.from_numpy(input_np).type(AppState().dtype)

    cnn = ConvInputModel()

    feature_maps = cnn(image)
    print('feature_maps:', feature_maps.shape)
    print('Computed output height, width:', cnn.get_output_shape(img_size, img_size))
