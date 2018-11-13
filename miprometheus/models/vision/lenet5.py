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
lenet5: a classical LeNet-5 model for MNIST digit classification. \
 To be taken as an illustrative example.
 """

__author__ = "Tomasz Kornuta & Vincent Marois"


import torch

# Import useful MI-Prometheus classes.
from miprometheus.models.model import Model


class LeNet5(Model):
    """
    A classical LeNet-5 model for MNIST digits classification. 
    """ 
    def __init__(self, params_, problem_default_values_):
        """
        Initializes the ``LeNet5`` model, creates the required layers.

        :param params: Parameters read from configuration file.
        :type params: ``miprometheus.utils.ParamInterface``

        :param problem_default_values_: dict of parameters values coming from the problem class.
        :type problem_default_values_: dict

        """
        super(LeNet5, self).__init__(params_, problem_default_values_)
        self.name = 'LeNet5'

        # Create the LeNet-5 layers.
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = torch.nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.linear1 = torch.nn.Linear(120, 84)
        self.linear2 = torch.nn.Linear(84, 10)

        # Create Model data definitions - indicate what a given model needs.
        self.data_definitions = {
            'images': {'size': [-1, 1, 32, 32], 'type': [torch.Tensor]}}

    def forward(self, data_dict):
        """
        Main forward pass of the ``LeNet5`` model.

        :param data_dict: DataDict({'images',**}), where:

            - images: [batch_size, num_channels, width, height]

        :type data_dict: ``miprometheus.utils.DataDict``

        :return: Predictions [batch_size, num_classes]

        """

        # Unpack DataDict.
        img = data_dict['images']

        # Pass inputs through layers.
        x = self.conv1(img)
        x = torch.nn.functional.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = x.view(-1, 120)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)

        return x  # return logits.
