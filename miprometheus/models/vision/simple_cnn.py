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
simple_cnn: a simple Convolutional Neural Network (CNN) designed specifically to solve MNIST and CIFAR 10 dataset. \
 To be taken as an illustrative example.
 """

__author__ = "Younes Bouhadjar & Vincent Marois"

import torch
import numpy as np
import torch.nn as nn

from miprometheus.models.model import Model


class SimpleConvNet(Model):
    """
    A simple 2 layers CNN designed specifically to solve ``MNIST`` & ``CIFAR10`` datasets. \
    The parameters here are not hardcoded so the user can adjust them for his application, \
    and see their impact on the model's behavior.
    """

    def __init__(self, params, problem_default_values_={}):
        """
        Constructor of the ``SimpleConvNet``. \

        The overall structure of this CNN is as follows:

            Conv1 -> MaxPool1 -> ReLu -> Conv2 -> MaxPool2 -> ReLu (-> flatten) -> Linear1 -> Linear2 -> Linear3

        The parameters that the user can change are:

         - For Conv1 & Conv2: number of output channels, kernel size, stride and padding.
         - For MaxPool1 & MaxPool2: Kernel size
         - For Linear3: The number of classes is read from ``problem_default_values_``. The number of output nodes for \
           Linear1 is set to 120, and Linear2 is fixed to 120 -> 84 for now. Linear3 is 84 -> nb_classes.


        .. note::

            We are using the default values of ``dilatation``, ``groups`` & ``bias`` for ``nn.Conv2D``.

            Similarly for the ``stride``, ``padding``, ``dilatation``, ``return_indices`` & ``ceil_mode`` of \
            ``nn.MaxPool2D``.


        The size of the images (width, height, number of channels) are read from ``problem_default_values_``. \
        Also, it is possible that the images are padded (with 0s) by the ``Problem`` class. The padding values \
        (e.g. [2,2,2,2]) should be indicated in ``problem_default_values_``, so that we can adjust the width & height.

        .. note::

            The images will be upscaled to [224, 224] (which is the input size of AlexNet, so this would \
            allow for comparison) if ``problem_default_values_['up_scaling']`` is ``True``.


        :param params: dict of parameters (read from configuration ``.yaml`` file).
        :type params: utils.ParamInterface

        :param problem_default_values_: default values coming from the ``Problem`` class.
        :type problem_default_values_: dict

        """
        # call base constructor.
        super(SimpleConvNet, self).__init__(params, problem_default_values_)

        # retrieve the Conv1 parameters
        self.out_channels_conv1 = params['conv1']['out_channels']
        self.kernel_size_conv1 = params['conv1']['kernel_size']
        self.stride_conv1 = params['conv1']['stride']
        self.padding_conv1 = params['conv1']['padding']

        # retrieve the Conv2 parameters
        self.out_channels_conv2 = params['conv2']['out_channels']
        self.kernel_size_conv2 = params['conv2']['kernel_size']
        self.stride_conv2 = params['conv2']['stride']
        self.padding_conv2 = params['conv2']['padding']

        # retrieve the MaxPool1 parameter
        self.kernel_size_maxpool1 = params['maxpool1']['kernel_size']

        # retrieve the MaxPool2 parameter
        self.kernel_size_maxpool2 = params['maxpool2']['kernel_size']

        # model name
        self.name = 'SimpleConvNet'

        # get image information from the problem class
        try:
            self.num_channels = problem_default_values_['num_channels']  # number of channels

            # upscale the image to [224, 224] if indicated.
            if problem_default_values_['up_scaling']:
                self.height = 224
                self.width = 224
                self.logger.warning('Upscaling the images to [224, 224].')
            else:
                self.height = problem_default_values_['height']
                self.width = problem_default_values_['width']

            # padding to use if specified.
            self.padding = problem_default_values_['padding']

            # number of output nodes
            self.nb_classes = problem_default_values_['nb_classes']

        except KeyError:
            self.logger.warning("Couldn't retrieve one or more value(s) from problem_default_values_.")

        # take into account the padding.
        self.height_padded = self.height + sum(self.padding[0:2])
        self.width_padded = self.width + sum(self.padding[2:4])

        self.data_definitions = {'images': {'size': [-1, self.num_channels, self.height_padded, self.width_padded],
                                            'type': [torch.Tensor]},
                                 'targets': {'size': [-1, 1], 'type': [torch.Tensor]}
                                 }

        # We can compute the spatial size of the output volume as a function of the input volume size (W),
        # the receptive field size of the Conv Layer neurons (F), the stride with which they are applied (S),
        # and the amount of zero padding used (P) on the border.
        # The corresponding equation is conv_size = ((W−F+2P)/S)+1.

        # doc for nn.Conv2D: https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
        # doc for nn.MaxPool2D: https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d

        # Conv1
        self.conv1 = nn.Conv2d(in_channels=self.num_channels,
                               out_channels=self.out_channels_conv1,
                               kernel_size=self.kernel_size_conv1,
                               stride=self.stride_conv1,
                               padding=self.padding_conv1,
                               dilation=1,
                               groups=1,
                               bias=True)

        self.width_features_conv1 = np.floor(
            ((self.width_padded - self.kernel_size_conv1 + 2*self.padding_conv1) / self.stride_conv1) + 1)
        self.height_features_conv1 = np.floor(
            ((self.height_padded - self.kernel_size_conv1 + 2*self.padding_conv1) / self.stride_conv1) + 1)

        # ----------------------------------------------------

        # MaxPool1
        self.maxpool1 = nn.MaxPool2d(kernel_size=self.kernel_size_maxpool1)

        self.width_features_maxpool1 = np.floor(
            ((self.width_features_conv1 - self.maxpool1.kernel_size + 2 * self.maxpool1.padding) / self.maxpool1.stride) + 1)

        self.height_features_maxpool1 = np.floor(
            ((self.height_features_conv1 - self.maxpool1.kernel_size + 2 * self.maxpool1.padding) / self.maxpool1.stride) + 1)

        # ----------------------------------------------------

        # Conv2
        self.conv2 = nn.Conv2d(in_channels=self.out_channels_conv1,
                               out_channels=self.out_channels_conv2,
                               kernel_size=self.kernel_size_conv2,
                               stride=self.stride_conv2,
                               padding=self.padding_conv2,
                               dilation=1,
                               groups=1,
                               bias=True)

        self.width_features_conv2 = np.floor(
            ((self.width_features_maxpool1 - self.kernel_size_conv2 + 2*self.padding_conv2) / self.stride_conv2) + 1)
        self.height_features_conv2 = np.floor(
            ((self.height_features_maxpool1 - self.kernel_size_conv2 + 2*self.padding_conv2) / self.stride_conv2) + 1)

        # ----------------------------------------------------

        # MaxPool2
        self.maxpool2 = nn.MaxPool2d(kernel_size=self.kernel_size_maxpool2)

        self.width_features_maxpool2 = np.floor(
            ((self.width_features_conv2 - self.maxpool2.kernel_size + 2 * self.maxpool2.padding) / self.maxpool2.stride) + 1)
        self.height_features_maxpool2 = np.floor(
            ((self.height_features_conv2 - self.maxpool2.kernel_size + 2 * self.maxpool2.padding) / self.maxpool2.stride) + 1)

        # ----------------------------------------------------

        # Linear layers

        self.linear1 = nn.Linear(in_features=self.out_channels_conv2 * self.width_features_maxpool2 * self.height_features_maxpool2,
                                 out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.linear3 = nn.Linear(in_features=84, out_features=self.nb_classes)

        # log some info.
        self.logger.info('Computed output shape of each layer:')
        self.logger.info('Input: [N, {}, {}, {}]'.format(self.num_channels, self.width_padded, self.height_padded))
        self.logger.info('Conv1: [N, {}, {}, {}]'.format(self.out_channels_conv1, self.width_features_conv1,
                                                      self.height_features_conv1))
        self.logger.info('MaxPool1: [N, {}, {}, {}]'.format(self.out_channels_conv1, self.width_features_maxpool1,
                                                      self.height_features_maxpool1))
        self.logger.info('Conv2: [N, {}, {}, {}]'.format(self.out_channels_conv2, self.width_features_conv2,
                                                      self.height_features_conv2))
        self.logger.info('MaxPool2: [N, {}, {}, {}]'.format(self.out_channels_conv2, self.width_features_maxpool2,
                                                         self.height_features_maxpool2))
        self.logger.info('Flatten: [N, {}]'.format(self.out_channels_conv2 * self.width_features_maxpool2 *
                                                   self.height_features_maxpool2))
        self.logger.info('Linear1: [N, {}]'.format(self.linear1.out_features))
        self.logger.info('Linear2: [N, {}]'.format(self.linear2.out_features))
        self.logger.info('Linear3: [N, {}]'.format(self.linear3.out_features))

        if self.app_state.visualize:
            self.output_conv1 = []
            self.output_conv2 = []

    def forward(self, data_dict):
        """
        forward pass of the ``SimpleConvNet`` model.

        :param data_dict: DataDict({'images','targets', 'targets_label'}), where:

            - images: [batch_size, num_channels, width, height],
            - targets [batch_size]

        :return: Predictions [batch_size, num_classes]

        """
        # get images
        images = data_dict['images']

        # apply Convolutional layer 1
        out_conv1 = self.conv1(images)
        if self.app_state.visualize:
            self.output_conv1 = out_conv1

        # apply max_pooling and relu
        out_maxpool1 = torch.nn.functional.relu(self.maxpool1(out_conv1))

        # apply Convolutional layer 2
        out_conv2 = self.conv2(out_maxpool1)
        if self.app_state.visualize:
            self.output_conv2 = out_conv2

        # apply max_pooling and relu
        out_maxpool2 = torch.nn.functional.relu(self.maxpool2(out_conv2))

        # flatten for the linear layers
        x = out_maxpool2.view(-1, self.out_channels_conv2 * self.width_features_maxpool2 * self.height_features_maxpool2)

        # apply 3 linear layers
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = self.linear3(x)

        return x

    def plot(self, data_dict, predictions, sample_number=0):
        """
        Simple plot - shows the ``Problem``'s images with the target & actual predicted class.\

        :param data_dict: DataDict({'images','targets', 'targets_label'})
        :type data_dict: utils.DataDict

        :param predictions: Predictions of the ``SimpleConvNet``.
        :type predictions: torch.tensor

        :param sample_number: Index of the sample in batch (DEFAULT: 0).
        :type sample_number: int

        """
        # Check if we are supposed to visualize at all.
        if not self.app_state.visualize:
            return False
        import matplotlib

        # unpack data_dict
        images = data_dict['images']
        targets = data_dict['targets']

        # Get sample.
        image = images[sample_number].cpu().detach().numpy()
        target = targets[sample_number].cpu().detach().numpy()
        prediction = predictions[sample_number].cpu().detach().numpy()

        # Reshape image.
        if image.shape[0] == 1:
            # This is single channel image - get rid of that dimension
            image = np.squeeze(image, axis=0)
        else:
            # More channels - move channels to axis2
            # (X : array_like, shape (n, m) or (n, m, 3) or (n, m, 4))
            image = image.transpose(1, 2, 0)

        # Show data.
        matplotlib.pyplot.title('Prediction: Class # {} (Target: Class # {})'.format(
            np.argmax(prediction), target))
        matplotlib.pyplot.imshow(image, interpolation='nearest', aspect='auto')

        # Show the feature maps of Conv1
        f1 = matplotlib.pyplot.figure()
        grid_size = int(np.sqrt(self.out_channels_conv1)) + 1
        gs = matplotlib.gridspec.GridSpec(grid_size, grid_size)

        for i in range(self.out_channels_conv1):
            ax = matplotlib.pyplot.subplot(gs[i])
            ax.imshow(self.output_conv1[0, i].detach().numpy())
        f1.suptitle('feature maps of Conv1')

        # Show the feature maps of Conv2
        f2 = matplotlib.pyplot.figure()
        grid_size = int(np.sqrt(self.out_channels_conv2)) + 1
        gs = matplotlib.gridspec.GridSpec(grid_size, grid_size)

        for i in range(self.out_channels_conv2):
            ax = matplotlib.pyplot.subplot(gs[i])
            ax.imshow(self.output_conv2[0, i].detach().numpy())
        f2.suptitle('feature maps of Conv2')

        # Plot!
        matplotlib.pyplot.show()


if __name__ == '__main__':
    # Set visualization.
    from miprometheus.utils.app_state import AppState
    AppState().visualize = True

    from miprometheus.utils.param_interface import ParamInterface
    from torch.utils.data import DataLoader
    from miprometheus.problems.image_to_class.mnist import MNIST

    problem_params = ParamInterface()
    problem_params.add_config_params({'use_train_data': True,
                                      'root_dir': '~/data/mnist',
                                      'padding': [0, 0, 0, 0],
                                      'up_scaling': False})
    batch_size = 64

    # create problem
    problem = MNIST(problem_params)
    print('Problem {} instantiated.'.format(problem.name))

    # instantiate DataLoader object
    dataloader = DataLoader(problem, batch_size=batch_size, collate_fn=problem.collate_fn)

    # Test base model.
    from miprometheus.utils.param_interface import ParamInterface
    model_params = ParamInterface()
    model_params.add_config_params({'conv1': {'out_channels': 6,
                                              'kernel_size': 5,
                                              'stride': 1,
                                              'padding': 0},
                                    'conv2': {'out_channels': 16,
                                              'kernel_size': 5,
                                              'stride': 1,
                                              'padding': 0},
                                    'maxpool1': {'kernel_size': 2},
                                    'maxpool2': {'kernel_size': 2}})

    # model
    model = SimpleConvNet(model_params, problem.default_values)
    print('Model {} instantiated.'.format(model.name))

    # perform handshaking between MAC & CLEVR
    model.handshake_definitions(problem.data_definitions)

    # generate a batch
    for i_batch, sample in enumerate(dataloader):
        print('Sample # {} - {}'.format(i_batch, sample['images'].shape), type(sample))
        logits = model(sample)
        print(logits.shape)

        # Plot it and check whether window was closed or not.
        if model.plot(sample, logits):
            break
