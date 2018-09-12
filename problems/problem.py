#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""problem.py: contains base class for all problems"""
__author__ = "Tomasz Kornuta & Vincent Marois"

import collections
from abc import abstractmethod

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from utils.app_state import AppState

import logging
logger = logging.Logger('DataDict')


class DataDict(collections.MutableMapping):
    """
    Mapping: A container object that supports arbitrary key lookups and implements the methods `__getitem__`, \
    `__iter__` and `__len__`.

    Mutable objects can change their value but keep their id() -> ease modifying existing keys' value.

    DataDict: Dict used for storing batches of data by problems.

    **This is the main object class used to share data between a problem class and a model class through a worker.**
    """

    def __init__(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)

    def __setitem__(self, key, value):
        #if key not in self.keys():
        #    logger.error('KeyError: Cannot modify a non-existing key.')
        #    raise KeyError('Cannot modify a non-existing key.')
        #else:
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        logger.error('KeyError: Not authorizing the deletion of a key.')
        raise KeyError('Not authorizing the deletion of a key.')

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        """
        :return: A simple dict representation of the mapping.
        """
        return str(self.__dict__)

    def __repr__(self):
        """
        Echoes class, id, & reproducible representation in the Read–Eval–Print Loop.
        """
        return '{}, DataDict({})'.format(super(DataDict, self).__repr__(), self.__dict__)

    def numpy(self):
        """
        Converts the DataDict to numpy objects.

        .. note::

            The torch.Tensor(s) contained in `self` are converted using `torch.Tensor.numpy()`: \
            This tensor and the returned ndarray share the same underlying storage. \
            Changes to self tensor will be reflected in the ndarray and vice versa.

            If an element of `self` is not a torch.Tensor, it is returned as is.


        :return: Converted DataDict.

        """
        numpy_datadict = self.__class__()

        for key in self:
            if isinstance(self[key], torch.Tensor):
                numpy_datadict[key] = self[key].numpy()
            else:
                numpy_datadict[key] = self[key]

        return numpy_datadict

    def cpu(self):
        """
        Moves the DataDict to memory accessible to the CPU.

        .. note::

            The torch.Tensor(s) contained in `self` are converted using torch.Tensor.cpu().
            If an element of `self` is not a torch.Tensor, it is returned as is, \
            i.e. We only move the torch.Tensor(s) contained in `self`. \


        :return: Converted DataDict.

        """
        cpu_datadict = self.__class__()

        for key in self:
            if isinstance(self[key], torch.Tensor):
                cpu_datadict[key] = self[key].cpu()
            else:
                cpu_datadict[key] = self[key]

        return cpu_datadict

    def cuda(self, device=None, non_blocking=False):
        """
        Returns a copy of this object in CUDA memory.

        .. note::

            Wraps call to torch.tensor.cuda(): If this object is already in CUDA memory and on the correct device, \
            then no copy is performed and the original object is returned.
            If an element of `self` is not a torch.Tensor, it is returned as is, \
            i.e. We only move the torch.Tensor(s) contained in `self`. \


        :param device: The destination GPU device. Defaults to the current CUDA device.
        :type device: torch.device

        :param non_blocking: If True and the source is in pinned memory, the copy will be asynchronous with respect to \
        the host. Otherwise, the argument has no effect. Default: False.
        :type non_blocking: bool

        """
        cuda_datadict = self.__class__()
        for key in self:
            if isinstance(self[key], torch.Tensor):
                cuda_datadict[key] = self[key].cuda(device=device, non_blocking=non_blocking)
            else:
                cuda_datadict[key] = self[key]

        return cuda_datadict

    def detach(self):
        """
        Returns a new DataDict, detached from the current graph.
        The result will never require gradient.

        .. note::
            Wraps call to `torch.Tensor.detach()`: the torch.Tensor(s) in the returned DataDict use the same
            data tensor(s) as the original one(s).
            In-place modifications on either of them will be seen, and may trigger errors in correctness checks.

        """
        detached_datadict = self.__class__()
        for key in self:
            if isinstance(self[key], torch.Tensor):
                detached_datadict[key] = self[key].detach()
            else:
                detached_datadict[key] = self[key]

        return detached_datadict


class Problem(Dataset):
    """
    Class representing base class for all Problems.
    Inherits from torch.utils.data.Dataset as all subclasses will represent a problem with an associated dataset.
    """

    def __init__(self, params):
        """
        Initializes problem object.

        :param params: Dictionary of parameters (read from configuration file).

        """
        # Store pointer to params.
        self.params = params

        # Set default loss function.
        self.loss_function = None


        # Size of the dataset
        self.length = None

        # data_definitions: this is used for defining the DataDict keys.
        # This dict contains information about the DataDict produced by the current problem class:
        # - e.g. for images, it can be {'images': {'width': 256, 'type': numpy.ndarray}}
        # - e.g. for sequences, it can be {'sequences': {'length': 10, 'type': torch.Tensor}}
        # etc.
        # This object will be used during handchecking between the model and the problem class to ensure that the model
        # can accept the batches produced by the problem.
        # This dict should at least contains the targets field.
        self.data_definitions = {'targets': {}}

        # Get access to AppState: for dtype, visualization flag etc.
        self.app_state = AppState()

        # "Default" problem name.
        self.name = 'Problem'

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


    @abstractmethod
    def collate_fn(self, batch):
        """
        Generates a batch of samples from a list of individuals samples retrieved by `__getitem__`.
        The default collate_fn is `torch.utils.data.default_collate`.

        .. note::

            **Abstract - to be defined in derived classes.**


        :param batch: Should be a list of DataDict retrieved by `__getitem__`, each containing tensors, numbers,
        dicts or lists.

        :return: DataDict containing the created batch.

        """
        return default_collate

    @abstractmethod
    def __getitem__(self, index):
        """
        Getter that returns an individual sample from the problem's associated dataset (that can be generated \
        on-the-fly, or retrieved from disk. It can also possibly be composed of several files.).

        To be redefined in subclasses.

        **The getter should return a DataDict: its keys should be defined by `self.data_definitions` keys.**

        e.g.:
            >>> data_dict = DataDict({key: None for key in self.data_definitions.keys()})
            >>> # you can now access each value by its key and assign the corresponding object (e.g. `torch.Tensor` etc)
            >>> ...
            >>> return data_dict



        .. warning::

            In a future version of `mi-prometheus`, multiprocessing will be supported for data loading.
            To construct a batch (say 64 samples), the indexes will be distributed among several workers (say 4, so that
            each worker has 16 samples to retrieve). It is best that samples can be accessed individually in the dataset
            folder so that there is no mutual exclusion between the workers and the performance is not degraded.

        :param index: index of the sample to return.

        :return: DataDict containing the sample.

        """

    def get_data_definitions(self):
        """
        Getter for the data_definitions dict so that it can be accessed by a worker to establish handchecking with
        the model class.

        :return: self.data_definitions()

        """
        return self.data_definitions

    def evaluate_loss(self, data_dict, logits):
        """
        Calculates loss between the predictions/logits and targets (from data_dict) using the selected loss function.

        :param data_dict: DataDict containing (among others) inputs and targets.
        :param logits: Predictions being output of the model.
        """

        # Compute loss using the provided loss function. 
        loss = self.loss_function(logits, data_dict['targets'])

        return loss

    def add_statistics(self, stat_col):
        """
        Adds statistics to collector.

        **EMPTY - To be redefined in inheriting classes.**

        :param stat_col: Statistics collector.

        """
        pass
        
    def collect_statistics(self, stat_col, data_dict, logits):
        """
        Base statistics collection.

        **EMPTY - To be redefined in inheriting classes.**
        The user has be ensure that the corresponding entry in the StatisticsCollector has been created in
        `add_statistics`.

        :param stat_col: Statistics collector.

        :param data_dict: DataDict containing inputs and targets.

        :param logits: Predictions being output of the model.

        """
        pass

    def get_epoch_size(self):
        """
        Compute the number of iterations ('episodes') to run given the size of the dataset and the batch size to cover
        the entire dataset once.

        .. note::

            We are counting the last batch, even though it might be smaller than the other ones if the size of the \
            dataset is not divisible by the batch size. -> Corresponds to drop_last=False in DataLoader().

        :return: Number of iterations to perform to go though the entire dataset once.

        """
        return self.length // self.params['batch_size'] + 1

    def initialize_epoch(self):
        """
        Function called to initialize a new epoch.

        The primary use is to reset statistics aggregators that track statistics over one epoch, e.g.:

            - Average accuracy over the epoch
            - Time taken for the epoch and average per batch
            - etc...

        **EMPTY - To be redefined in inheriting classes.**

        """
        pass

    def finalize_epoch(self):
        """
        Function called at the end of an epoch to execute a few tasks, e.g.:

            - Compute the mean accuracy over the epoch
            - Get the time taken for the epoch and per batch
        This function will use the statistics aggregators set up (or reset) in `initialize_epoch()`.

        **EMPTY - To be redefined in inheriting classes.**

        .. note::

            TODO: To display the final results for the current epoch, this function should use the Logger.


        """
        pass

    def plot_preprocessing(self, data_dict, logits):
        """
        Allows for some data preprocessing before the model creates a plot for visualization during training or
        inference.

        **EMPTY - To be redefined in inheriting classes.**


        :param data_dict: DataDict.

        :param logits: Logits being output of the model.

        :return: data_tuple, logits after preprocessing.

        """
        return data_dict, logits

    def curriculum_learning_initialize(self, curriculum_params):
        """
        Initializes curriculum learning - simply saves the curriculum params.
        This method can be overwritten in the derived classes.

        :param curriculum_params: Interface to parameters accessing curriculum learning view of the registry tree.
        """
        # Save params.
        self.curriculum_params = curriculum_params

    def curriculum_learning_update_params(self, episode):
        """
        Updates problem parameters according to curriculum learning.
        There is no general solution to curriculum learning.
        This method should be overwritten in the derived classes.

        :param episode: Number of the current episode.
        :return: True informing that CL wasn't active at all (i.e. is finished).
        """
        return True


if __name__ == '__main__':
    """Unit test for DataDict"""

    data_definitions = {'inputs': {'size': [64, 20], 'type': int}, 'targets': {'size': [64], 'type': int}}

    datadict = DataDict({key: None for key in data_definitions.keys()})

    #datadict['inputs'] = torch.ones([64, 20, 512]).type(torch.FloatTensor)
    #datadict['targets'] = torch.ones([64, 20]).type(torch.FloatTensor)

    print(repr(datadict))
