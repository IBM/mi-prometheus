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
Image_encoder.py: Implementation of the ImageEncoder for the VWM network.

"""

__author__ = "Vincent Albouy, T.S. Jayram"

from torch.nn import Module
import torch.nn as nn

class ImageEncoder(Module):

    """
    Implementation of the ``ImageEncoder`` of the VWM network.
    """

    def __init__(self, dim):
        """
        Constructor for the ``ImageEncoder``.

        :param dim: dimension of feature vectors
        :type dim: int

        """

        # call base constructor
        super(ImageEncoder, self).__init__()

        def cnn_layer(in_channels, out_channels, post_process=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, 3, stride=1,
                          padding=1, dilation=1, groups=1, bias=True),
                nn.MaxPool2d(2, stride=None, padding=0, dilation=1,
                             return_indices=False, ceil_mode=False),
                nn.BatchNorm2d(out_channels, affine=post_process)]
            if post_process:
                layers.append(nn.ReLU())

            return layers

        # Number of channels in input Image RGB
        image_channels = 3  # [R,G,B]

        # Visual processing channels across cnn layers
        # Hard coded for COG dataset
        vp_channels = [32, 64, 64]

        layer_list = cnn_layer(image_channels, vp_channels[0])

        layer_list.extend(cnn_layer(vp_channels[0], vp_channels[1]))
        layer_list.extend(cnn_layer(vp_channels[1], vp_channels[2]))

        layer_list.extend(cnn_layer(vp_channels[2], dim, post_process=False))

        self.cnn_module = nn.Sequential(*layer_list)

        def init_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                m.bias.data.fill_(0.01)

        self.cnn_module.apply(init_weights)

    def forward(self, images):
        """
        Forward pass of the ``ImageEncoder``.

        :param images: tensor of the images, shape [batch_size x H x W].

        :return feature_maps: [batch_size x (H*W) x dim].

        """
        x = self.cnn_module(images)

        # reshape from 4D to 3D and permute so that embedding dimension is last
        feature_maps = x.flatten(start_dim=-2).transpose(-1, -2)

        # optional
        feature_maps = feature_maps.contiguous()

        # return feature_maps
        return feature_maps
