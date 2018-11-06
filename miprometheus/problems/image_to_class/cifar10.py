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
cifar10.py: contains code for loading the `CIFAR10` dataset using ``torchvision``.
"""
__author__ = "Younes Bouhadjar & Vincent Marois"

import os
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms


from miprometheus.utils.data_dict import DataDict
from miprometheus.problems.image_to_class.image_to_class_problem import ImageToClassProblem


class CIFAR10(ImageToClassProblem):
    """
    Classic CIFAR10 classification problem.

    Please see reference here: https://www.cs.toronto.edu/~kriz/cifar.html

    .. warning::

        The dataset is not originally split into a training set, validation set and test set; only \
        training and test set. It is recommended to use a validation set.

        ``torch.utils.data.SubsetRandomSampler`` is recommended.

    """

    def __init__(self, params):
        """
        Initializes CIFAR10 problem:

            - Calls ``problems.problem.ImageToClassProblem`` class constructor,
            - Sets following attributes using the provided ``params``:

                - ``self.data_folder`` (`string`) : Root directory of dataset where the directory \
                 ``cifar-10-batches-py`` will be saved,
                - ``self.use_train_data`` (`bool`, `optional`) : If ``True``, creates dataset from training set, \
                    otherwise creates from test set,
                - ``self.resize`` : (optional) resize the images to `[h, w]` if set,
                - ``self.defaut_values`` :

                    >>> self.default_values = {'num_classes': 10,
                    >>>            'num_channels': 3,
                    >>>            'width': self.width, # (DEFAULT: 32)
                    >>>            'height': self.height} # DEFAULT: 32)

                - ``self.data_definitions`` :

                    >>> self.data_definitions = {'images': {'size': [-1, 3, self.height, self.width], 'type': [torch.Tensor]},
                    >>>                          'targets': {'size': [-1], 'type': [torch.Tensor]},
                    >>>                          'targets_label': {'size': [-1, 1], 'type': [list, str]}
                    >>>                         }

        .. warning::

            Resizing images might cause a significant slow down in batch generation.

        .. note::

            The following is set by default:

            >>> params = {'data_folder': '~/data/cifar10',
            >>>           'use_train_data': True}


        :param params: Dictionary of parameters (read from configuration ``.yaml``file).
        :type params: miprometheus.utils.ParamInterface

        """

        # Call base class constructors.
        super(CIFAR10, self).__init__(params, 'CIFAR10')
        
        # Set default parameters.
        params.add_default_params({'data_folder': '~/data/cifar10',
                                   'use_train_data': True})

        # Get absolute path.
        data_folder = os.path.expanduser(params['data_folder'])

        # Retrieve parameters from the dictionary.
        self.use_train_data = params['use_train_data']

        # Add transformations depending on the resizing option.
        if ('resize' in self.params):
            # Check the desired size.
            if len(self.params['resize']) != 2:
                self.logger.error("'resize' field must contain 2 values: the desired height and width")
                exit(-1)

            # Output image dimensions.
            self.height = self.params['resize'][0]
            self.width = self.params['resize'][1]
            self.num_channels = 3

            # Up-scale and transform to tensors.
            transform = transforms.Compose([transforms.Resize((self.height, self.width)), transforms.ToTensor()])

            self.logger.warning('Upscaling the images to [{}, {}]. Slows down batch generation.'.format(
                self.width, self.height))

        else:
            # Default MNIST settings.
            self.width = 32
            self.height = 32
            self.num_channels = 3
            # Simply turn to tensor.
            transform = transforms.Compose([transforms.ToTensor()])

        # Define the default_values dict: holds parameters values that a model may need.
        self.default_values = {'num_classes': 10,
                               'num_channels': self.num_channels,
                               'width': self.width,
                               'height': self.height}


        self.data_definitions = {'images': {'size': [-1, self.num_channels, self.height, self.width], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1], 'type': [torch.Tensor]},
                                 'targets_label': {'size': [-1, 1], 'type': [list, str]}
                                 }

        # load the dataset
        self.dataset = datasets.CIFAR10(root=data_folder, train=self.use_train_data,
                                        download=True, transform=transform)
        # type(self.train_dataset) = <class 'torchvision.datasets.cifar.CIFAR10'>
        # -> inherits from torch.utils.data.Dataset

        self.length = len(self.dataset)
        # Class names.
        self.labels = 'Airplane Automobile Bird Cat Deer Dog Frog Horse Shipe Truck'.split(' ')

    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a sample.

        :param index: index of the sample to return.
        :type index: int

        :return: ``DataDict({'images','targets', 'targets_label'})``, with:

            - images: Image, upscaled if ``self.up_scaling`` and pad if ``self.padding``,
            - targets: Index of the target class
            - targets_label: Label of the target class (cf ``self.labels``)


        """

        img, target = self.dataset.__getitem__(index)
        target = torch.tensor(target)

        label = self.labels[target.data]

        data_dict = DataDict({key: None for key in self.data_definitions.keys()})
        data_dict['images'] = img
        data_dict['targets'] = target
        data_dict['targets_label'] = label

        return data_dict

    def collate_fn(self, batch):
        """
        Combines a list of ``DataDict`` (retrieved with ``__getitem__`` ) into a batch.

        .. note::

            This function wraps a call to ``default_collate`` and simply returns the batch as a ``DataDict``\
            instead of a dict.

            Multi-processing is supported as the data sources are small enough to be kept in memory\
            (`self.root-dir/cifar-10-batches/data_batch_i` have a size of 31.0 MB).

        :param batch: list of individual ``DataDict`` samples to combine.

        :return: ``DataDict({'images','targets', 'targets_label'})`` containing the batch.

        """

        return DataDict({key: value for key, value in zip(self.data_definitions.keys(),
                                                          super(CIFAR10, self).collate_fn(batch).values())})


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # set the seeds
    np.random.seed(0)
    torch.manual_seed(0)

    # Load parameters.
    from miprometheus.utils.param_interface import ParamInterface
    params = ParamInterface()  # using the default values

    batch_size = 64

    # Create problem.
    cifar10 = CIFAR10(params)

    # get a sample
    sample = cifar10[0]
    print('__getitem__ works.\n')

    # wrap DataLoader on top of this Dataset subclass
    from torch.utils.data.dataloader import DataLoader

    dataloader = DataLoader(dataset=cifar10, collate_fn=cifar10.collate_fn,
                            batch_size=batch_size, shuffle=True, num_workers=0)

    # try to see if there is a speed up when generating batches w/ multiple workers
    import time
    s = time.time()
    for i, batch in enumerate(dataloader):
        print('Batch # {} - {}'.format(i, type(batch)))
    print('Number of workers: {}'.format(dataloader.num_workers))
    print('time taken to exhaust the dataset for a batch size of {}: {}s'.format(batch_size, time.time() - s))

    # Display single sample (0) from batch.
    batch = next(iter(dataloader))
    cifar10.show_sample(batch, 0)

    print('Unit test completed')
