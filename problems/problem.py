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

"""problem.py: contains base class for all problems"""
__author__ = "Tomasz Kornuta & Vincent Marois"

import numpy as np
import logging

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from utils.app_state import AppState
from utils.data_dict import DataDict



class Problem(Dataset):
    """
    Class representing base class for all Problems.

    Inherits from torch.utils.data.Dataset as all subclasses will represent a problem with an associated dataset,\
    and the `worker` will use ``torch.utils.data.dataloader.DataLoader`` to generate batches.

    Implements features & attributes used by all subclasses.

    """

    def __init__(self, params):
        """
        Initializes problem object.

        :param params: Dictionary of parameters (read from the configuration ``.yaml`` file).

        This constructor:

        - stores a pointer to ``params``:

            >>> self.params = params

        - sets a default loss function:

            >>> self.loss_function = None

        - initializes the size of the dataset:

            >>> self.length = None

        - sets a default problem name:

            >>> self.name = 'Problem'

        - initializes the logger.

            >>> self.logger = logging.Logger(self.name)

        - initializes the data definitions: this is used for defining the ``DataDict`` keys.

        .. note::

            This dict contains information about the DataDict produced by the current problem class.

            This object will be used during handshaking between the model and the problem class to ensure that the model
            can accept the batches produced by the problem.

            This dict should at least contains the `targets` field:

                >>> self.data_definitions = {'targets': {'size': [-1, 1], 'type': [torch.Tensor]}}

        - initializes the default values: this is used to pass missing parameters values to the model.

        .. note::

            It is likely to encounter a case where the model needs a parameter value only known when the problem has been
            instantiated, like the size of a vocabulary set or the number of marker bits.

            The user can fill in those values in this dict, which will be passed to the model in its  `__init__`  . The
            model will then be able to fill it its missing parameters values, either from params or this dict.

                >>> self.default_values = {}

        - sets the access to ``AppState``: for dtype, visualization flag etc.

            >>> self.app_state = AppState()

        """
        # Store pointer to params.
        self.params = params
        # Empty curriculum learning params - for now.
        self.curriculum_params = {}

        # Set default loss function.
        self.loss_function = None

        # Size of the dataset
        self.length = None

        # "Default" problem name.
        self.name = 'Problem'

        # initialize the logger.
        self.logger = logging.Logger(self.name)

        # data_definitions: this is used for defining the DataDict keys.

        # This dict contains information about the DataDict produced by the current problem class.
        # This object will be used during handshaking between the model and the problem class to ensure that the model
        # can accept the batches produced by the problem.
        # This dict should at least contains the `targets` field.
        self.data_definitions = {'targets': {'size': [-1, 1], 'type': [torch.Tensor]}}

        # default_values: this is used to pass missing parameters values to the model.

        # It is likely to encounter a case where the model needs a parameter value only known when the problem has been
        # instantiated, like the size of a vocabulary set or the number of marker bits.
        # The user can fill in those values in this dict, which will be passed to the model in its  `__init__`  . The
        # model will then be able to fill it its missing parameters values, either from params or this dict.
        self.default_values = {}

        # Get access to AppState: for dtype, visualization flag etc.
        self.app_state = AppState()

    def handshake_definitions(self, model_data_definitions_):
        """
        Proceeds to the handshake between what the Model produces as predictions and what the Problem expects to compute\
        the loss

        .. note::

            Handshaking is defined here as making sure that the ``Model`` and the ``Problem`` agree on the data that they
            exchange.
            More specifically, the ``Problem`` has a definition of the targets data that it expects\
            (through its ``self.data_definitions`` attribute). The ``Model`` has the same object describing what \
            it generates.

            This functions proceeds to the handshaking as:

                - Verifying that the key ``targets`` is present in ``Model.data_definitions`` (representing the logits)\
                and in ``Problem.data_definitions`` (representing the ground truth answers).
                If not, an exception is thrown.

                - If this key exists, than this function checks that the shape and type specified in\
                   ``Model.data_definitions`` are accepted by the loss function specified by the Problem.
                   Similarly, it also checks if the shape & type indicated for ``Problem.data_definitions_``\
                    are accepted by its loss function. If not, an exception is thrown.


            **If both steps above passed, than the Problem accepts what the Model generates as predictions.**


            To properly define the ``data_definitions`` dicts, here are some examples:

                >>> data_definitions = {'img': {'size': [-1, 320, 480, 3], 'type': [np.ndarray]},
                >>>                     'question': {'size': [-1, -1], 'type': [torch.Tensor]},
                >>>                     'question_length': {'size': [-1], 'type': [list, int]},
                >>>                     # ...
                >>>                     }

                Please indicate both the size and the type as ``lists``:

                    - Indicate all dimensions in the correct order for each key `size` field. If a dimension is\
                    unimportant or unknown (e.g. the batch size or variable-length sequences), then please indicate \
                    ``-1`` at the correct location. Also indicate the corect number of dimensions.

                    - both the ground truth targets and the logits should be ``torch.tensor``.


        :param model_data_definitions_: Contains the definition of the logits generated by the ``Model`` class (among\
        other definitions).
        :type model_data_definitions_: dict

        :return: True if the loss function can accept the logits and ground truth labels produced by the ``Model``\
         and ``Problem`` respectively, otherwise throws an exception.

        """

        if 'targets' in self.data_definitions.keys() and 'targets' in model_data_definitions_.keys():

            # check the type first, easier
            if self.data_definitions['targets']['type'] == [torch.Tensor]\
                    and model_data_definitions_['targets']['type'] == [torch.Tensor]:

                if type(self.loss_function).__name__ in ['L1Loss', 'MSELoss', 'PoissonNLLLoss', 'KLDivLoss', 'BCELoss',
                                                         'BCEWithLogitsLoss', 'HingeEmbeddingLoss',
                                                         'MultiLabelMarginLoss', 'SmoothL1Loss', 'SoftMarginLoss',
                                                         'MultiLabelSoftMarginLoss', 'MaskedBCEWithLogitsLoss']:

                    # these loss functions require the same shape for both the logits and ground truth labels
                    if len(self.data_definitions['targets']['size']) != len(model_data_definitions_['targets']['size']):
                        # the ground truth labels and the logits don't have the same number of dimensions
                        raise ValueError("Both the logits and ground truth labels don't have the same number of "
                                         "dimensions. The specified loss function ({}) requires it.".format(self.loss_function))
                    else:
                        # both have same number of dim, now check that the indicated dimensions are equal
                        # checking that even the -1 are in the same place.
                        for i, dim in enumerate(self.data_definitions['targets']['size']):
                                if dim != model_data_definitions_['targets']['size'][i]:
                                    raise ValueError('The specified loss function ({}) require that the logits '
                                                     'and ground truth labels have the same shape. Got logits shape = {}'
                                                     ' and ground truth labels shape = {}.'.format(self.loss_function,
                                                                                                  model_data_definitions_['targets']['size'],
                                                                                                  self.data_definitions['targets']['size']))
                # these loss functions require that the ground truth labels have 1 less dimension
                elif type(self.loss_function).__name__ in ['CrossEntropyLoss', 'NLLLoss', 'MarginRankingLoss',
                                                           'MultiMarginLoss', 'MaskedCrossEntropyLoss']:

                    if len(self.data_definitions['targets']['size']) != len(model_data_definitions_['targets']['size'])-1:
                        # the ground truth labels and the logits don't have the same number of dimensions
                        raise ValueError("The specified loss function ({}) requires that the ground truth labels"
                                         " have one less dimension than the logits. Got logits shape = {}"
                                         " and ground truth labels shape = {}.".format(self.loss_function,
                                                                                       model_data_definitions_['targets']['size'],
                                                                                       self.data_definitions['targets']['size']))
                    # TODO: should also check that the order of dimension is coherent

                else:
                    self.logger.warning('The indicated loss function is {}, which requires more than 2 inputs.'
                                        ' Not checking it for now.'.format(self.loss_function))

            else:
                raise ValueError("Either the logits or ground truth labels are not torch.Tensor. ")

        else:
            raise KeyError("Couldn't find the key 'targets' in self.data_definitions or model_data_definitions_.")

        # Everything matches, return true
        return True

    def __len__(self):
        """
        :return: The size of the dataset.

        """
        return self.length

    def set_loss_function(self, loss_function):
        """
        Sets loss function.

        :param loss_function: Loss function (e.g. nn.CrossEntropyLoss()) that will be set as the optimization criterion.
        """
        self.loss_function = loss_function

    def collate_fn(self, batch):
        """
        Generates a batch of samples from a list of individuals samples retrieved by ``__getitem__``.

        The default collate_fn is ``torch.utils.data.default_collate``.

        .. note::

            This base ``collate_fn`` method only calls the default ``torch.utils.data.default_collate``\
            , as it can handle several cases (mainly tensors, numbers, dicts and lists).

            If your dataset can yield variable-length samples within a batch, or generate batches `on-the-fly`\
            , or possesses another ''non regular'' characteristic, it is most likely that you will need to \
            override this default ``collate_fn``.


        :param batch: Should be a list of DataDict retrieved by `__getitem__`, each containing tensors, numbers,\
        dicts or lists.

        :return: DataDict containing the created batch.

        """
        return default_collate(batch)

    def __getitem__(self, index):
        """
        Getter that returns an individual sample from the problem's associated dataset (that can be generated \
        `on-the-fly`, or retrieved from disk. It can also possibly be composed of several files.).

        .. note::

            **To be redefined in subclasses.**


        .. note::

            **The getter should return a DataDict: its keys should be defined by** ``self.data_definitions`` **keys.**

            This ensures consistency of the content of the ``DataDict`` when processing to the ``handshake``\
            between the ``Problem`` class and the ``Model`` class. For more information, please see\
             ``models.model.Model.handshake_definitions``.

            e.g.:

                >>> data_dict = DataDict({key: None for key in self.data_definitions.keys()})
                >>> # you can now access each value by its key and assign the corresponding object (e.g. `torch.tensor` etc)
                >>> ...
                >>> return data_dict



        .. warning::

            `Mi-Prometheus` supports multiprocessing for data loading (through the use of\
             ``torch.utils.data.dataloader.DataLoader``).

            To construct a batch (say 64 samples), the indexes are distributed among several workers (say 4, so that
            each worker has 16 samples to retrieve). It is best that samples can be accessed individually in the dataset
            folder so that there is no mutual exclusion between the workers and the performance is not degraded.

            If each sample is generated `on-the-fly`, this shouldn't cause a problem. There may be an issue with \
            randomness. Please refer to the official Pytorch documentation for this.


        :param index: index of the sample to return.
        :type index: int

        :return: Empty ``DataDict``, having the same key as ``self.data_definitions``.

        """
        return DataDict({key: None for key in self.data_definitions.keys()})

    def worker_init_fn(self, worker_id):
        """
        Function to be called by ``torch.utils.data.dataloader.DataLoader`` on each worker subprocess, \
        after seeding and before data loading. (default: ``None``).

        .. note::

            Set the ``NumPy`` random seed of the worker equal to the previous NumPy seed + its ``worker_id``\
             to avoid having all workers returning the same random numbers.


        :param worker_id: the worker id (in [0, ``torch.utils.data.dataloader.DataLoader.num_workers`` - 1])
        :type worker_id: int

        :return: ``None`` by default
        """
        np.random.seed(seed=np.random.get_state()[1][0] + worker_id)

    def get_data_definitions(self):
        """
        Getter for the data_definitions dict so that it can be accessed by a ``worker`` to establish handshaking with
        the ``Model`` class.

        :return: self.data_definitions()

        """
        return self.data_definitions

    def evaluate_loss(self, data_dict, logits):
        """
        Calculates loss between the predictions/logits and targets (from data_dict) using the selected loss function.

        :param data_dict: DataDict containing (among others) inputs and targets.
        :type data_dict: DataDict

        :param logits: Predictions of the model.

        :return: Loss.
        """

        # Compute loss using the provided loss function. 
        loss = self.loss_function(logits, data_dict['targets'])

        return loss

    def add_statistics(self, stat_col):
        """
        Adds statistics to ``StatisticsCollector``.

        .. note::


            Empty - To be redefined in inheriting classes.


        :param stat_col: ``StatisticsCollector``.

        """
        pass
        
    def collect_statistics(self, stat_col, data_dict, logits):
        """
        Base statistics collection.

         .. note::


            Empty - To be redefined in inheriting classes. The user has to ensure that the corresponding entry \
            in the ``StatisticsCollector`` has been created with ``self.add_statistics()`` beforehand.

        :param stat_col: ``StatisticsCollector``.

        :param data_dict: ``DataDict`` containing inputs and targets.
        :type data_dict: DataDict

        :param logits: Predictions being output of the model.

        """
        pass

    def add_aggregators(self, stat_agg):
        """
        Adds statistical aggregators to ``StatisticsAggregator``.

        .. note::

            Empty - To be redefined in inheriting classes.


        :param stat_agg: ``StatisticsAggregator``.

        """
        pass

    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates the statistics collected by ''StatisticsCollector'' and adds the results to ''StatisticsAggregator''.

         .. note::

            Only computes the min, max, mean, std of the loss as these are basic statistical aggregator \
            set by default.

            The user can override this function in subclasses but should call \
            ``super().collect_aggregator(stat_col, stat_agg)`` to collect basic statistical aggregator.

            Given that the ``StatisticsAggregator`` uses the statistics collected by the ``StatisticsCollector``, \
            the user should also ensure that these statistics are correctly collected \
            (i.e. use of ``self.add_statistics`` and ``self.collect_statistics``.

        :param stat_col: ``StatisticsCollector``.

        :param stat_agg: ``StatisticsAggregator``.

        """
        pass
        
    def get_epoch_size(self, batch_size):
        """
        Compute the number of iterations ('episodes') to run given the size of the dataset and the batch size to cover
        the entire dataset once.

        :param batch_size: Batch size.
        :type batch_size: int

        .. note::

            We are counting the last batch, even though it might be smaller than the other ones if the size of the \
            dataset is not divisible by the batch size. -> Corresponds to ``drop_last=False`` in ``DataLoader()``.

        :return: Number of iterations to perform to go though the entire dataset once.

        """
        if (self.length % batch_size) == 0:
            return self.length // batch_size
        else:
            return (self.length // batch_size) + 1

    def initialize_epoch(self, epoch):
        """
        Function called to initialize a new epoch.

        The primary use is to reset ``StatisticsAggregators`` that track statistics over one epoch, e.g.:

            - Average accuracy over the epoch
            - Time taken for the epoch and average per batch
            - etc...

        .. note::


            Empty - To be redefined in inheriting classes.

        :param epoch: current epoch index
        :type epoch: int


        """
        pass

    def finalize_epoch(self, epoch):
        """
        Function called at the end of an epoch to execute a few tasks, e.g.:

            - Compute the mean accuracy over the epoch,
            - Get the time taken for the epoch and per batch
            - etc.

        This function will use the ``StatisticsAggregators`` set up (or reset) in ``self.initialize_epoch()`.

        .. note::


            Empty - To be redefined in inheriting classes.

            TODO: To display the final results for the current epoch, this function should use the Logger.

        :param epoch: current epoch index
        :type epoch: int

        """
        pass

    def plot_preprocessing(self, data_dict, logits):
        """
        Allows for some data preprocessing before the model creates a plot for visualization during training or
        inference.

        .. note::


            Empty - To be redefined in inheriting classes.


        :param data_dict: ``DataDict``.
        :type data_dict: DataDict

        :param logits: Predictions of the model.

        :return: data_dict, logits after preprocessing.

        """
        return data_dict, logits

    def curriculum_learning_initialize(self, curriculum_params):
        """
        Initializes curriculum learning - simply saves the curriculum params.

        .. note::

            This method can be overwritten in the derived classes.


        :param curriculum_params: Interface to parameters accessing curriculum learning view of the registry tree.
        """
        # Save params.
        self.curriculum_params = curriculum_params

    def curriculum_learning_update_params(self, episode):
        """
        Updates problem parameters according to curriculum learning.

        .. note::

            This method can be overwritten in the derived classes.

        :param episode: Number of the current episode.
        :type episode: int

        :return: True informing that Curriculum Learning wasn't active at all (i.e. is finished).

        """

        return True

