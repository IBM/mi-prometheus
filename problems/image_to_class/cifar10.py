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

"""cifar10.py: contains code of loading CIFAR10 dataset using torchvision"""
__author__ = "Younes Bouhadjar"

import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F

from problems.problem import DataDict
from problems.image_to_class.image_to_class_problem import ImageToClassProblem


class CIFAR10(ImageToClassProblem):
    """
    Classic CFIAR10 classification problem.

    TODO: DOCUMENTATION.
    """

    def __init__(self, params):
        """
        Initializes CIFAR problem, calls base class initialization, sets
        properties using the provided parameters.

        TODO: DOCUMENTATION.

        :param params: Dictionary of parameters (read from configuration file).

        """

        # Call base class constructors.
        super(CIFAR10, self).__init__(params)

        # Retrieve parameters from the dictionary.

        self.use_train_data = params['use_train_data']
        self.root_dir = params['root_dir']

        # possibility to pad the image
        self.padding = params['padding']

        # up scaling the image to 224, 224 if True
        self.up_scaling = params['up_scaling']

        # define the default_values dict: holds parameters values that a model may need.
        self.default_values = {'nb_classes': 10,
                               'num_channels': 3,
                               'width': 32,
                               'height': 32,
                               'up_scaling': self.up_scaling,
                               'padding': self.padding}

        self.height = 224 if self.up_scaling else self.default_values['height']
        self.width = 224 if self.up_scaling else self.default_values['width']

        self.data_definitions = {'images': {'size': [-1, 3, self.height, self.width], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1, 1], 'type': [torch.Tensor]},
                                 'targets_label': {'size': [-1, 1], 'type': [list, str]}
                                 }

        self.name = 'CIFAR10'

        # Define transforms: takes in an PIL image and returns a transformed version
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(
        )]) if self.up_scaling else transforms.Compose([transforms.ToTensor()])

        # load the dataset
        self.dataset = datasets.CIFAR10(root=self.root_dir, train=self.use_train_data,
                                        download=True, transform=transform)
        # type(self.train_dataset) = <class 'torchvision.datasets.cifar.CIFAR10'>
        # -> inherits from torch.utils.data.Dataset

        self.length = len(self.dataset)

        # Class names.
        self.labels = 'Airplane Automobile Bird Cat Deer Dog Frog Horse Shipe Truck'.split(
            ' ')

    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a sample.

        :param index: index of the sample to return.

        :return: DataDict({'images','targets', 'targets_label'}), with:

            - images: Image, transformed if a transform is specified in ``__init__``.
            - targets: Index of the target class
            - targets_label: Label of the target class (cf ``self.labels``)


        """

        img, target = self.dataset.__getitem__(index)
        target = torch.tensor(target)

        # pad img
        img = F.pad(img, self.padding, 'constant', 0)

        label = self.labels[target.data]

        data_dict = DataDict({key: None for key in self.data_definitions.keys()})
        data_dict['images'] = img
        data_dict['targets'] = target
        data_dict['targets_label'] = label

        return data_dict

    def collate_fn(self, batch):
        """
        Combines a list of DataDict (retrieved with __getitem__) into a batch.

        .. note::

            This function wraps a call to ``default_collate`` and simply returns the batch as a DataDict\
            instead of a dict.
            Multi-processing is supported as the data sources are small enough to be kept in memory\
            (`training.pt` has a size of 47.5 MB).

        :param batch: list of individual ``DataDict`` samples to combine.

        :return: DataDict({'images','targets', 'targets_label'}) containing the batch.

        """

        return DataDict({key: value for key, value in zip(self.data_definitions.keys(),
                                                          super(CIFAR10, self).collate_fn(batch).values())})




if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""
    np.random.seed(0)
    torch.manual_seed(0)

    # "Loaded parameters".
    from utils.param_interface import ParamInterface 
    params = ParamInterface()
    params.add_default_params({
        'batch_size': 64,
        'use_train_data': True,
        'folder': '~/data/cifar10',
        'padding': [0, 0, 0, 0],
        'up_scaling': True})

    # Create problem object.
    cifar10 = CIFAR10(params)

    sample = cifar10[0]
    print(repr(sample))
    print('__getitem__ works.')

    # wrap DataLoader on top of this Dataset subclass
    from torch.utils.data.dataloader import DataLoader

    dataloader = DataLoader(dataset=cifar10, collate_fn=cifar10.collate_fn,
                            batch_size=params['batch_size'], shuffle=True, num_workers=4)

    # try to see if there is a speed up when generating batches w/ multiple workers

    # HAVE TO MANAGE INDEXES FOR THE TRAINING / VALIDATION SPLIT

    import time
    s = time.time()
    for i, batch in enumerate(dataloader):
        print('Batch # {} - {}'.format(i, type(batch)))

    print('Number of workers: {}'.format(dataloader.num_workers))
    print('time taken to exhaust the dataset for a batch size of {}: {}s'.format(params['batch_size'], time.time() - s))

    # Display single sample (0) from batch.
    #batch = next(iter(dataloader))
    #cifar10.show_sample(batch, 0)
