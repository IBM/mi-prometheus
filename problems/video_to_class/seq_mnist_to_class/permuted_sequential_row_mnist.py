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

"""permuted_sequential_row_mnist.py: load MNIST dataset using torchvision and apply a permutation over the rows"""
__author__ = "Younes Bouhadjar"

import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from problems.problem import DataTuple, MaskAuxTuple
from problems.video_to_class.video_to_class_problem import VideoToClassProblem


class PermutedSequentialRowMnist(VideoToClassProblem):
    """
    Class generating permuted sequences of rows for sequential mnist.
    """

    def __init__(self, params):
        """
        Constructor for  PermutedSequentialRowMnist.

        :param params: dictionary of parameters

        """

        # Call base class constructors.
        super(PermutedSequentialRowMnist, self).__init__(params)

        # Retrieve parameters from the dictionary.
        self.batch_size = params['batch_size']
        self.start_index = params['start_index']
        self.stop_index = params['stop_index']
        self.num_rows = 28
        self.num_columns = 28
        self.use_train_data = params['use_train_data']
        self.datasets_folder = params['mnist_folder']

    def generate_batch(self):
        # define transforms
        pixel_permutation = torch.randperm(28)
        train_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Lambda(lambda x: x[:, pixel_permutation])])

        # load the datasets
        train_datasets = datasets.MNIST(
            self.datasets_folder,
            train=True,
            download=True,
            transform=train_transform)
        # set split
        num_train = len(train_datasets)
        indices = list(range(num_train))

        idx = indices[self.start_index: self.stop_index]
        sampler = SubsetRandomSampler(idx)

        # loader
        train_loader = torch.utils.data.DataLoader(
            train_datasets, batch_size=self.batch_size, sampler=sampler)
        # create an iterator
        train_loader = iter(train_loader)

        # create mask
        mask = torch.zeros(self.num_rows)
        mask[-1] = 1

        # train_loader a generator: (data, label)
        (data, label) = next(train_loader)

        # Return DataTuple(!) and an empty (aux) tuple.
        return DataTuple(data, label), MaskAuxTuple(mask.type(torch.uint8))


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # "Loaded parameters".
    params = {'batch_size': 1, 'start_index': 0, 'stop_index': 54999,
              'use_train_data': True, 'mnist_folder': '~/data/mnist'}
    # Create problem object.
    problem = PermutedSequentialRowMnist(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    num_rows = 28
    num_columns = 28
    sample = 0
    data_tuple, _ = next(generator)
    x, y = data_tuple

    # Display single sample (0) from batch.
    problem.show_sample(x[sample, 0], y)
