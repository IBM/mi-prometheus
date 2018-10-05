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

"""simple_cnn: a simple Convolutional neural network designed specifically to solve MNIST and CIFAR 10 dataset """

__author__ = "Younes Bouhadjar"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import Model
from problems.problem import DataDict


class SimpleConvNet(Model):
    """
    A simple Convolutional neural network designed specifically to solve MNIST
    and CIFAR 10 dataset, The parameters here are not hardcoded so the user can
    adjust them for his application.
    """

    def __init__(self, params, problem_default_values_={}):
        super(SimpleConvNet, self).__init__(params)
        """
        Constructor of the SimpleConvNet

        :param: dictionary of parameters
        """

        # retrieve parameters from the yaml file
        # cnn parameters
        self.depth_conv1 = params['depth_conv1']
        self.depth_conv2 = params['depth_conv2']
        self.filter_size_conv1 = params['filter_size_conv1']
        self.filter_size_conv2 = params['filter_size_conv2']
        self.num_pooling = params['num_pooling']

        self.name = 'SimpleConvNet'

        # get image informations from the problem class
        try:
            # image size
            self.num_channels = problem_default_values_['num_channels']
            self.height = 224 if problem_default_values_['up_scaling'] else problem_default_values_['height']
            self.width = 224 if problem_default_values_['up_scaling'] else problem_default_values_['width']
            self.padding = problem_default_values_['padding']

            # number of output neurons
            self.nb_classes = problem_default_values_['nb_classes']

        except BaseException:
            self.logger.warning("Couldn't retrieve one or more value(s) from problem_default_values_.")

        self.data_definitions = {'images': {'size': [-1, self.num_channels, self.height, self.width], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1, 1], 'type': [torch.Tensor]}
                                 }

        self.height_padded = self.height + sum(self.padding[0:2])
        self.width_padded = self.width + sum(self.padding[2:4])

        # Input size of the first fully connected layer:
        # We can compute the spatial size of the output volume as a function of the input volume size (W),
        # the receptive field size of the Conv Layer neurons (F), the stride with which they are applied (S),
        # and the amount of zero padding used (P) on the border. You can convince yourself that the correct formula
        # for calculating how many neurons “fit” is given by  (W−F+2P)/S+1.

        # TODO: for now we assume that padding = 0 and stride = 1
        self.width_features_conv1 = np.floor(
            ((self.width_padded - self.filter_size_conv1) + 1) / self.num_pooling)
        self.height_features_conv1 = np.floor(
            ((self.height_padded - self.filter_size_conv1) + 1) / self.num_pooling)

        self.width_features_conv2 = np.floor(
            ((self.width_features_conv1 - self.filter_size_conv2) + 1) / self.num_pooling)
        self.height_features_conv2 = np.floor(
            ((self.height_features_conv1 - self.filter_size_conv2) + 1) / self.num_pooling)

        self.conv1 = nn.Conv2d(
            self.num_channels,
            self.depth_conv1,
            kernel_size=self.filter_size_conv1)
        self.conv2 = nn.Conv2d(
            self.depth_conv1,
            self.depth_conv2,
            kernel_size=self.filter_size_conv2)

        self.fc1 = nn.Linear(
            self.depth_conv2 *
            self.width_features_conv2 *
            self.height_features_conv2,
            120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.nb_classes)

        if self.app_state.visualize:
            self.output_conv1 = []
            self.output_conv2 = []

    def forward(self, data_dict):
        """
        forward pass of SimpleConvNet model.

        :param data_dict: DataDict({'images','targets', 'targets_label'}), where:

            - images: [batch_size, num_channels, width, height],
            - targets [batch_size]

        :return: Predictions [batch_size, num_classes]

        """
        # unpack data_dict
        images, _, _ = data_dict.values()

        # apply Convolutional layer 1
        x1 = self.conv1(images)
        if self.app_state.visualize:
            self.output_conv1 = x1

        # apply max_pooling and relu
        x1_max_pool = F.relu(F.max_pool2d(x1, self.num_pooling))

        # apply Convolutional layer 2
        x2 = self.conv2(x1_max_pool)
        if self.app_state.visualize:
            self.output_conv2 = x2

        # apply max_pooling and relu
        x2_max_pool = F.relu(F.max_pool2d(x2, self.num_pooling))

        x = x2_max_pool.view(-1, self.depth_conv2 *
                             self.width_features_conv2 * self.height_features_conv2)

        # apply 3 linear layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def plot(self, data_dict, predictions, sample_number=0):
        """
        Simple plot - shows MNIST image with target and actual predicted class.

        :param data_dict: DataDict({'images','targets', 'targets_label'})
        :param predictions: Prediction.
        :param sample_number: Number of sample in batch (DEFAULT: 0)
        """
        # Check if we are supposed to visualize at all.
        if not self.app_state.visualize:
            return False
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # unpack data_dict
        images, targets, targets_label = data_dict.values()

        # Get sample.
        image = images[sample_number].cpu().detach().numpy()
        target = targets[sample_number].cpu().detach().numpy()
        prediction = predictions[sample_number].cpu().detach().numpy()

        # Reshape image.
        if image.shape[0] == 1:
            # This is single channel image - get rid of that dimension
            image = np.squeeze(image, axis=0)
        else:
            # More channels - move channels to axis2, according to matplotilb doc it should be ok
            # (X : array_like, shape (n, m) or (n, m, 3) or (n, m, 4))
            image = image.transpose(1, 2, 0)

        # Show data.
        plt.title('Prediction: {} (Target: {})'.format(
            np.argmax(prediction), target))
        plt.imshow(image, interpolation='nearest', aspect='auto')

        # feature layer 2
        f = plt.figure()
        grid_size = int(np.sqrt(self.depth_conv1)) + 1
        gs = gridspec.GridSpec(grid_size, grid_size)

        for i in range(self.depth_conv1):
            ax = plt.subplot(gs[i])
            ax.imshow(self.output_conv1[0, i].detach().numpy())

        # feature layer 1
        f = plt.figure()
        grid_size = int(np.sqrt(self.depth_conv2)) + 1
        gs = gridspec.GridSpec(grid_size, grid_size)

        for i in range(self.depth_conv2):
            ax = plt.subplot(gs[i])
            ax.imshow(self.output_conv2[0, i].detach().numpy())

        # Plot!
        plt.show()
        exit()


if __name__ == '__main__':
    # Set visualization.
    from utils.app_state import AppState
    AppState().visualize = True

    from utils.param_interface import ParamInterface
    from torch.utils.data.dataloader import DataLoader
    from problems.image_to_class.mnist import MNIST
    from problems.image_to_class.cifar10 import CIFAR10

    problem_params = ParamInterface()
    problem_params.add_custom_params({'use_train_data': True,
                                      'root_dir': '~/data/cifar10',
                                      'padding': [0, 0, 0, 0],
                                      'up_scaling': False})
    batch_size = 64

    # create problem
    dataset = CIFAR10(problem_params)
    print('Problem {} instantiated.'.format(dataset.name))

    # instantiate DataLoader object
    problem = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

    # Test base model.
    from utils.param_interface import ParamInterface
    model_params = ParamInterface()
    model_params.add_custom_params({
        'depth_conv1': 10,
        'depth_conv2': 20,
        'filter_size_conv1': 5,
        'filter_size_conv2': 5,
        'num_pooling': 2})

    # model
    model = SimpleConvNet(model_params, dataset.default_values)
    print('Model {} instantiated.'.format(model.name))

    # perform handshaking between MAC & CLEVR
    model.handshake_definitions(dataset.data_definitions)

    # generate a batch
    for i_batch, sample in enumerate(problem):
        print('Sample # {} - {}'.format(i_batch, sample['images'].shape), type(sample))
        logits = model(sample)
        print(logits.shape)

        # Plot it and check whether window was closed or not.
        #if model.plot(sample, logits):
        #    break
