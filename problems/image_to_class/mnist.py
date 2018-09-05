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

"""mnist.py: contains code of loading MNIST dataset using torchvision"""
__author__ = "Younes Bouhadjar"

import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

from problems.problem import DataTuple, LabelAuxTuple
from problems.image_to_class.image_to_class_problem import ImageToClassProblem


class MNIST(ImageToClassProblem):
    """
    Classic MNIST classification problem.
    """

    def __init__(self, params):
        """
        Initializes MNIST problem, calls base class initialization, sets properties using the provided parameters.

        :param params: Dictionary of parameters (read from configuration file).
        """

        # Call base class constructors.
        super(MNIST, self).__init__(params)

        # Retrieve parameters from the dictionary.
        self.batch_size = params['batch_size']
        self.start_index = params['start_index']
        self.stop_index = params['stop_index']
        self.use_train_data = params['use_train_data']
        self.datasets_folder = params['mnist_folder']
        self.padding = params['padding']
        # up scaling the image to 224, 224 if True
        self.up_scaling = params['up_scaling']

        # Define transforms
        train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(
        )]) if self.up_scaling else transforms.Compose([transforms.ToTensor()])

        # load the datasets
        self.train_datasets = datasets.MNIST(
            self.datasets_folder,
            train=self.use_train_data,
            download=True,
            transform=train_transform)

        # set split data (for training and validation data)
        num_train = len(self.train_datasets)
        indices = list(range(num_train))
        idx = indices[self.start_index: self.stop_index]
        self.sampler = SubsetRandomSampler(idx)

        # Class names.
        self.mnist_class_names = 'Zero One Two Three Four Five Six Seven Eight Nine'.split(
            ' ')

    def generate_batch(self):

        # data loader
        train_loader = torch.utils.data.DataLoader(
            self.train_datasets,
            batch_size=self.batch_size,
            sampler=self.sampler)
        # create an iterator
        train_loader = iter(train_loader)

        # train_loader a generator: (data, label)
        (data, label) = next(train_loader)

        # padding data
        data_padded = F.pad(data, self.padding, 'constant', 0)

        # Generate labels for aux tuple
        class_names = [self.mnist_class_names[i] for i in label]

        # Return DataTuple(!) and an empty (aux) tuple.
        return DataTuple(data_padded, label), LabelAuxTuple(class_names)


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    params = {
        'batch_size': 2,
        'start_index': 0,
        'stop_index': 54999,
        'use_train_data': True,
        'mnist_folder': '~/data/mnist',
        'padding': [
            4,
            4,
            3,
            3],
        'up_scaling': False}

    # Create problem object.
    problem = MNIST(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    dt, at = next(generator)

    # Display single sample (0) from batch.
    problem.show_sample(dt, at, 0)
