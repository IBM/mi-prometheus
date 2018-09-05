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

"""image_encoding.py: image encoding for VQA problem, same as in this paper https://arxiv.org/abs/1706.01427, specifically desinged for sort of clevr """
__author__ = "Younes Bouhadjar"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ImageEncoding(nn.Module):
    """
    Image encoding using 4 convolutional layers with batch normalization, it was designed specifically for sort of clevr https://arxiv.org/abs/1706.01427
    """

    def __init__(self):
        """
        Constructor of the ImageEncoding class
        """

        super(ImageEncoding, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.batchNorm4 = nn.BatchNorm2d(256)

    def forward(self, img):
        """
        Apply 4 convolutional layers over the image

        :param img: input image [batch_size, num_channels, height, width]

        :return x: feature map with flattening the width and height dimensions to a single one and transpose it with the num_channel dimension
          [batch_size, new_height * new_width, num_channels_encoded_question]

        """
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)

        # flattening the width and height dimensions to a single one and
        # transpose it with the num_channel dimension, necessary when applying
        # the attention
        x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)

        return x


class PretrainedImageEncoding(nn.Module):
    """
    Image encoding using pretrained resnetXX from torchvision
    """

    def __init__(self, cnn_model='resnet18', num_blocks=2):
        """
        Constructor of the PretrainedImageEncoding class

        :param cnn_model: select which resnet pretrained model to load
        :param num_blocks: num of resnet blocks to be used
        """

        super(PretrainedImageEncoding, self).__init__()

        num_blocks = num_blocks

        # Get resnet18
        cnn = getattr(torchvision.models, cnn_model)(pretrained=True)

        # First layer added with num_channel equal 3
        layers = [
            cnn.conv1,
            cnn.bn1,
            cnn.relu,
            cnn.maxpool,
        ]

        # select the resnet blocks and append them to layers
        for i in range(num_blocks):
            name = 'layer%d' % (i + 1)
            layers.append(getattr(cnn, name))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, img):
        """
        Apply a pretrained cnn

        :param img: input image [batch_size, num_channels, height, width]

        :return x: feature map with flattening the width and height dimensions to a single one and transpose it with the num_channel dimension
          [batch_size, new_height * new_width, num_channels_encoded_question]

        """
        # Apply model image encoding
        x = self.model(img)

        # flattening the width and height dimensions to a single one and
        # transpose it with the num_channel dimension, necessary when applying
        # the attention
        x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)

        return x
