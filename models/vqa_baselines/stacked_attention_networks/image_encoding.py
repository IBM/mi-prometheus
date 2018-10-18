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
image_encoding.py: Contains a class using a pretrained CNN from ``torchvision`` as the image encoding \
for the Stacked Attention Network.

"""
__author__ = "Vincent Marois & Younes Bouhadjar"

import torch
import torch.nn as nn
import torchvision


class PretrainedImageEncoding(nn.Module):
    """
    Wrapper class over a ``torchvision.model`` to produce feature maps for the SAN model.

    """

    def __init__(self, cnn_model='resnet18', num_layers=2):
        """
        Constructor of the ``PretrainedImageEncoding`` class.

        :param cnn_model: select which pretrained model to load.
        :type cnn_model: str

        .. warning::

            This class has only been tested with the ``resnet18`` model.

        :param num_layers: Number of layers to select from the ``cnn_model``.
        :type num_layers: int


        """
        # call base constructor
        super(PretrainedImageEncoding, self).__init__()

        # Get pretrained cnn model
        self.cnn_model = cnn_model
        cnn = getattr(torchvision.models, self.cnn_model)(pretrained=True)

        # First layer added with num_channel equal 3
        layers = [
            cnn.conv1,
            cnn.bn1,
            cnn.relu,
            cnn.maxpool,
        ]

        # select the following layers and append them.
        for i in range(1, num_layers+1):
            name = 'layer%d' % i
            layers.append(getattr(cnn, name))

        self.model = torch.nn.Sequential(*layers)

    def get_output_nb_filters(self):
        """
        :return: The number of filters of the last conv layer.
        """
        try:
            nb_channels = self.model[-1][-1].bn2.num_features
            return nb_channels
        except:
            print('Could not get the number of output channels of the model {}'.format(self.cnn_model))

    def forward(self, img):
        """
        Forward pass of a pretrained cnn model.

        :param img: input image [batch_size, num_channels, height, width]
        :type img: torch.tensor

        :return x: feature maps, [batch_size, output_channels, new_height, new_width]

        """
        # Apply model image encoding
        return self.model(img)


if __name__ == '__main__':

    img_encoding = PretrainedImageEncoding()
    print(img_encoding.get_output_nb_filters())
