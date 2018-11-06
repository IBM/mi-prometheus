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
alexnet_wrapper.py: Contains a small wrapper class to the AlexNet model available in ``TorchVision``.

"""
__author__ = "Tomasz Kornuta, Younes Bouhadjar, Vincent Marois"

import torch
import numpy as np
from torchvision.models import alexnet

from miprometheus.models.model import Model


class AlexnetWrapper(Model):
    """
    Wrapper class to Alexnet model from TorchVision.
    """

    def __init__(self, params, problem_default_values_={}):
        """
        Constructor for the AlexNet wrapper. Simply instantiate the Alexnet model \
        from ``torchvision.models.``

        .. note::

           The model expects input images normalized as follows: \
           mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least `224`. \
           The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] \
           and std = [0.229, 0.224, 0.225].


        :param params: dictionary of parameters (read from the ``.yaml`` configuration file.)

        :param problem_default_values_: default values coming from the ``Problem`` class.
        :type problem_default_values_: dict

        """
        # call base constructor
        super(AlexnetWrapper, self).__init__(params, problem_default_values_)

        try:
            # number of output nodes
            self.nb_classes = problem_default_values_['nb_classes']

        except KeyError:
            self.logger.warning("Couldn't retrieve the number of classes from problem_default_values_.")

        # set model from torchvision
        self.model = alexnet(pretrained=params['pretrained'], num_classes=self.nb_classes)

        self.name = 'AlexNetWrapper'

        self.data_definitions = {'images': {'size': [-1, -1, 224, 224],
                                            'type': [torch.Tensor]},
                                 'targets': {'size': [-1, 1], 'type': [torch.Tensor]}
                                 }

    def forward(self, data_dict):
        """
        Main forward pass of the Alexnet wrapper.

        :param data_dict: DataDict({'images',**}), where:

            - images: [batch_size, num_channels, width, height],

        :return: Predictions [batch_size, num_classes]
        """

        images = data_dict['images']

        # checks if the num_channels is different than 3 (e.g. for MNIST)
        if images.size(1) != 3:
            # inputs_size = (batch_size, num_channel, numb_columns, num_rows)
            num_channel = 3
            inputs_size = (images.size(0), num_channel, images.size(2), images.size(3))
            inputs = torch.zeros(inputs_size).type(self.app_state.dtype)

            for i in range(num_channel):
                inputs[:, None, i, :, :] = images

            # pass the transformed images through the model
            outputs = self.model(inputs)

        else:
            # pass directly the images through the model
            outputs = self.model(images)

        return outputs

    def plot(self, data_dict, predictions, sample_number=0):
        """
        Simple plot - shows the ``Problem``'s images with the target & actual predicted class.\

        :param data_dict: DataDict({'images','targets', 'targets_label'})
        :type data_dict: utils.DataDict

        :param predictions: Predictions of the ``AlexnetWrapper``.
        :type predictions: torch.tensor

        :param sample_number: Index of the sample in batch (DEFAULT: 0).
        :type sample_number: int

        """
        # Check if we are supposed to visualize at all.
        if not self.app_state.visualize:
            return False
        import matplotlib.pyplot as plt

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
        plt.title('Prediction: Class # {} (Target: Class # {})'.format(
            np.argmax(prediction), target))
        plt.imshow(image, interpolation='nearest', aspect='auto')

        # Plot!
        plt.show()


if __name__ == '__main__':
    # Set visualization.
    from miprometheus.utils.app_state import AppState
    AppState().visualize = True

    from miprometheus.utils.param_interface import ParamInterface
    from torch.utils.data import DataLoader
    from miprometheus.problems import CIFAR10

    problem_params = ParamInterface()
    problem_params.add_config_params({'use_train_data': True,
                                      'root_dir': '~/data/cifar10',
                                      'padding': [0, 0, 0, 0],
                                      'up_scaling': True})
    batch_size = 64

    # create problem
    problem = CIFAR10(problem_params)
    print('Problem {} instantiated.'.format(problem.name))

    # instantiate DataLoader object
    dataloader = DataLoader(problem, batch_size=batch_size, collate_fn=problem.collate_fn)

    # Test base model.
    from miprometheus.utils.param_interface import ParamInterface
    model_params = ParamInterface()
    model_params.add_config_params({'pretrained': False})

    # model
    model = AlexnetWrapper(model_params, problem.default_values)
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
