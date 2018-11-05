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

"""permuted_sequential_row_mnist.py: loads the `MNIST` dataset using ``torchvision`` and\
 apply a permutation over the rows"""
__author__ = "Younes Bouhadjar & Vincent Marois"

import torch
from torchvision import datasets, transforms

from miprometheus.utils.data_dict import DataDict
from miprometheus.problems.video_to_class.video_to_class_problem import VideoToClassProblem


class PermutedSequentialRowMnist(VideoToClassProblem):
    """
    The Permuted MNIST is a sequence of classification tasks in which the rows\
     of the input images are swapped with a random permutation.

    .. warning::

        The dataset is not originally split into a training set, validation set and test set; only\
        training and test set. It is recommended to use a validation set.

        ``torch.utils.data.SubsetRandomSampler`` is recommended.

    """

    def __init__(self, params):
        """
        Initializes PermutedSequentialRowMnist problem:

            - Calls ``problems.problem.VideoToClassProblem`` class constructor,
            - Sets following attributes using the provided ``params``:

                - ``self.root_dir`` (`string`) : Root directory of dataset where ``processed/training.pt``\
                    and  ``processed/test.pt`` will be saved,
                - ``self.use_train_data`` (`bool`, `optional`) : If True, creates dataset from ``training.pt``,\
                    otherwise from ``test.pt``
                - ``self.defaut_values`` :

                    >>> self.default_values = {'nb_classes': 10,
                    >>>                        'num_channels': 1,
                    >>>                        'width': 28,
                    >>>                        'height': 28}

                - ``self.data_definitions`` :

                    >>> self.data_definitions = {'images': {'size': [-1, 3, -1, -1], 'type': [torch.Tensor]},
                    >>>                          'mask': {'size': [-1, -1, -1, -1], 'type': [torch.Tensor]},
                    >>>                          'targets': {'size': [-1, 1], 'type': [torch.Tensor]},
                    >>>                          'targets_label': {'size': [-1, 1], 'type': [list, str]}
                    >>>                         }

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).

        """

        # Call base class constructor.
        super(PermutedSequentialRowMnist, self).__init__(params)

        # Retrieve parameters from the dictionary.
        self.use_train_data = params['use_train_data']
        self.root_dir = params['root_dir']

        self.num_rows = 28
        self.num_columns = 28

        # define the default_values dict: holds parameters values that a model may need.
        self.default_values = {'nb_classes': 10,
                               'num_channels': 1,
                               'width': 28,
                               'height': 28,
                               }

        self.data_definitions = {'images': {'size': [-1, 1, 28, 28], 'type': [torch.Tensor]},
                                 'mask': {'size': [-1, 1], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1, 1], 'type': [torch.Tensor]},
                                 'targets_label': {'size': [-1, 1], 'type': [list, str]}
                                 }

        self.name = 'PermutedSequentialRowMNIST'

        # Class names.
        self.labels = 'Zero One Two Three Four Five Six Seven Eight Nine'.split(' ')

        # define transforms
        pixel_permutation = torch.randperm(self.num_rows)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x[:, pixel_permutation])])

        # load the dataset
        self.dataset = datasets.MNIST(self.root_dir, train=self.use_train_data,
                                      download=True, transform=transform)

        self.length = len(self.dataset)

    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a sample.

        :param index: index of the sample to return.
        :type index: int

        :return: ``DataDict({'images','targets', 'targets_label'})``, with:

            - images: Image,
            - mask,
            - targets: Index of the target class
            - targets_label: Label of the target class (cf ``self.labels``)


        """
        # get sample
        img, target = self.dataset.__getitem__(index)

        # get label
        label = self.labels[target.data]

        # create mask
        mask = torch.zeros(self.num_rows).type(self.app_state.IntTensor)
        mask[-1] = 1

        data_dict = DataDict({key: None for key in self.data_definitions.keys()})
        data_dict['images'] = img
        data_dict['mask'] = mask
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
            (`training.pt` has a size of 47.5 MB).

        :param batch: list of individual ``DataDict`` samples to combine.

        :return: ``DataDict({'images','targets', 'targets_label'})`` containing the batch.

        """

        return DataDict({key: value for key, value in zip(self.data_definitions.keys(),
                                                          super(PermutedSequentialRowMnist, self).collate_fn(batch).values())})


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # Load parameters.
    from miprometheus.utils.param_interface import ParamInterface
    params = ParamInterface()
    params.add_default_params({'use_train_data': True, 'root_dir': '~/data/mnist'})

    batch_size = 64

    # Create problem.
    problem = PermutedSequentialRowMnist(params)

    # get a sample
    sample = problem[0]
    print(repr(sample))
    print('__getitem__ works.')

    # wrap DataLoader on top of this Dataset subclass
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset=problem, collate_fn=problem.collate_fn,
                            batch_size=batch_size, shuffle=True, num_workers=8)

    # try to see if there is a speed up when generating batches w/ multiple workers
    import time

    s = time.time()
    for i, batch in enumerate(dataloader):
        print('Batch # {} - {}'.format(i, type(batch)))
        print(batch['images'].shape)
        break

    print('Number of workers: {}'.format(dataloader.num_workers))
    print('time taken to exhaust the dataset for a batch size of {}: {}s'.format(batch_size, time.time() - s))

    # Display single sample (0) from batch.
    batch = next(iter(dataloader))
    problem.show_sample(batch, 0)

    print('Unit test completed')
