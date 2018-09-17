#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
##
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
generate_feature_maps.py: This file contains 1 class:

    - GenerateFeatureMaps: This class instantiates a specified pretrained CNN model to extract feature maps from\
     images stored in the indicated directory. It also creates a DataLoader to generate batches of these images.

This class is used in problems.image_text_to_class.CLEVR.generate_feature_maps_file.

"""
__author__ = "Vincent Marois"
import os
import torchvision
from torchvision import transforms
import torch
from PIL import Image

from torch.utils.data import Dataset


class GenerateFeatureMaps(Dataset):
    """
    Class handling the generation of feature using a pretrained CNN for specified images.
    """

    def __init__(self, image_dir, cnn_model, num_blocks, filename_template, set='train', transform=transforms.ToTensor):
        """
        Creates the pretrained CNN model & move it to CUDA if available.

        :param image_dir: Directory path to the images to extract features from.
        :type image_dir: str

        :param cnn_model: Name of the pretrained CNN model to use. Must be in ``torchvision.models.``
        :type cnn_model: str

        :param num_blocks: number of layers to use from the cnn_model. **This is dependent on the specified\
        cnn_model, please check this value beforehand.**

        :param filename_template: The template followed by the filenames in ``image_dir``. It should indicate with\
         brackets where the index is located, e.g.

            >>> filename_template = 'CLEVR_train_{}.png'

        The index will be filled up on 6 characters.

        :param set: The dataset split to use. e.g. ``train``, ``val`` etc.
        :type set: str, optional.

        :param transform: ``torchvision.transform`` to apply on the images before passing them to the CNN model.\
        default:

            >>> transform = transforms.ToTensor

        :type transform: transforms, optional.

        """
        # call base constructor
        super(GenerateFeatureMaps, self).__init__()

        # parse params
        self.image_dir = image_dir
        self.set = set
        self.cnn_model = cnn_model
        self.num_blocks = num_blocks
        self.transform = transform
        self.filename_template = filename_template

        # Get specified pretrained cnn model
        cnn = getattr(torchvision.models, self.cnn_model)(pretrained=True)

        # First layer added with num_channel equal 3
        layers = [
            cnn.conv1,
            cnn.bn1,
            cnn.relu,
            cnn.maxpool,
        ]

        # get subsequent layers: May not work for all torchvision.models!
        for i in range(1, self.num_blocks):
            name = 'layer%d' % i
            layers.append(getattr(cnn, name))

        # build pretrained cnn cut at specified layer
        self.model = torch.nn.Sequential(*layers)

        # move it to CUDA & specify evaluation behavior
        self.model.cuda() if torch.cuda.is_available() else None
        self.model.eval()

        # set the dataset size as the numbers of images in the folder
        self.length = len(os.listdir(os.path.expanduser(self.image_dir)))

    def __getitem__(self, index):
        """
        Gets a image from the ``image_dir`` and apply a transform on it if specified.

        :param index: index of the sample to get.

        :return: transformed image as a tensor (shape should be [224, 224, 3])

        """
        # open image
        img = os.path.join(self.image_dir, self.filename_template.format(str(index).zfill(6)))
        img = Image.open(img).convert('RGB')

        # apply transform & return it as a tensor.
        return self.transform(img)

    def __len__(self):
        """
        :return: length of dataset.
        """
        return self.length
