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

"""conv_input_model.py: contains CNN model for the Relational Network."""
__author__ = "Vincent Marois"

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.app_state import AppState

class ConvInputModel(nn.Module):
    """
    Simple 4 layers CNN for image encoding in the Relational Network model.
    """

    def __init__(self):

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

    def forward(self, img):
        """
        Forward pass of the CNN.
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
    Unit Test for the ConvInputModel.
    """

    # "Image" - batch x channels x width x height
    batch_size = 64
    img_size = 128

    input_np = np.random.binomial(1, 0.5, (batch_size, 3, img_size, img_size))
    image = torch.from_numpy(input_np).type(AppState().dtype)

    cnn = ConvInputModel()

    feature_maps = cnn(image)
    print('feature_maps:', feature_maps.shape)
