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

"""generate-feature_maps.py: This file contains 1 class:

          - GenerateFeatureMaps: This class instantiates a specified pretrained CNN model to extract feature maps from
          images stored in the indicated directory. It also creates a DataLoader to generate batches of these images.
          This class is used in problems.image_text_to_class.new_clevr_dataset.generate_feature_maps_file.
  """
__author__ = "Vincent Marois"
import torchvision
from torchvision import transforms
import torch
from PIL import Image

from torch.utils.data import Dataset

# Add path to main project directory
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..', '..'))

from misc.app_state import AppState
app_state = AppState()


class GenerateFeatureMaps(Dataset):
    """
    Class handling the generation of feature using a pretrained CNN for the images of the CLEVR dataset.
    """

    def __init__(self, clevr_dir, set, cnn_model='resnet101', num_blocks=4):
        """
        Creates the pretrained CNN model & move it to CUDA.
        WARNING: WE ONLY AUTHORIZE USING THIS CLASS ON GPU, TO SPEED UP THE FORWARD PASSES OF THE PRETRAINED CNN.

        :param clevr_dir: Directory path to the CLEVR dataset.
        :param set: String to specify which dataset to use: 'train', 'val' or 'test'.
        :param cnn_model: pretrained CNN model to use
        :param num_blocks: number of layers to use from the cnn_model.
        """
        # we only authorize the images processing on GPU.
        if not torch.cuda.is_available():
            print('\nWARNING: CUDA IS NOT AVAILABLE. NOT AUTHORIZING DO FEATURE MAPS EXTRACTION. EXITING.')
            exit(1)

        # call base constructor
        super(GenerateFeatureMaps, self).__init__()

        # parse params
        self.clevr_dir = clevr_dir
        self.set = set
        self.cnn_model = cnn_model
        self.num_blocks = num_blocks

        # Get specified pretrained cnn model
        cnn = getattr(torchvision.models, self.cnn_model)(pretrained=True)

        # First layer added with num_channel equal 3
        layers = [
            cnn.conv1,
            cnn.bn1,
            cnn.relu,
            cnn.maxpool,
        ]

        # get subsequent layers
        for i in range(1, self.num_blocks):
            name = 'layer%d' % i
            layers.append(getattr(cnn, name))

        # build pretrained cnn cut at specified layer
        self.model = torch.nn.Sequential(*layers)

        # move it to CUDA & specify evaluation behavior
        self.model.cuda()
        self.model.eval()

        self.length = len(os.listdir(os.path.join(self.clevr_dir, 'images', self.set)))

        self.transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        """
        Gets a image from the CLEVR directory and apply a transform on it.
        :param index: index of the sample to get.

        :return: transformed image as a tensor (shape should be [224, 224, 3])
        """
        # open image
        img = os.path.join(self.clevr_dir, 'images', self.set, 'CLEVR_{}_{}.png'.format(self.set, str(index).zfill(6)))
        img = Image.open(img).convert('RGB')

        # apply transform & return it as a tensor.
        return self.transform(img)

    def __len__(self):
        """
        :return: length of dataset.
        """
        return self.length
