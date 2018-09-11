#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""problem.py: contains base class for all problems"""
__author__ = "Tomasz Kornuta & Vincent Marois"

import collections
from abc import abstractmethod

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

import utils
print(utils.__file__)
from utils.app_state import AppState

import logging
logger = logging.Logger('DataDict')


class DataDict(collections.MutableMapping):
    """
    Mapping: A container object that supports arbitrary key lookups and implements the methods __getitem__, __iter__ \
    and __len__.
    Mutable objects can change their value but keep their id() -> ease modifying existing keys' value.
    DataDict: Dict used for storing batches of data by problems.
    """

    def __init__(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)

    def __setitem__(self, key, value):
        if key not in self.keys():
            logger.error('KeyError: Cannot modify a non-existing key.')
            raise KeyError('Cannot modify a non-existing key.')
        else:
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
        Convert the content of the DataDict to numpy objects (if possible).

        TODO: Not sure if the following is respected with this implementation here:
            - This tensor and the returned ndarray share the same underlying storage. \
              Changes to self tensor will be reflected in the ndarray and vice versa.

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
        In-place function.
        """
        for key in self:
            if isinstance(self[key], torch.Tensor):
                self[key] = self[key].cpu()

    def cuda(self, device=None, non_blocking=False):
        """
        Returns a copy of this object in CUDA memory.

        Wraps call to torch.tensor.cuda(): If this object is already in CUDA memory and on the correct device, then no
        copy is performed and the original object is returned.

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
        Returns a new Tensor, detached from the current graph.
        The result will never require gradient.

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
        # Set default loss function.
        self.loss_function = None

        # Set default collate function.
        self.collate_fn = default_collate

        # Size of the dataset
        self.length = None

        # Store pointer to params.
        self.params = params

        # data definition: this is used for defining the DataDict keys
        self.data_definition = {}

        # Get access to AppState.
        self.app_state = AppState()

        # "Default" problem name.
        self.name = 'Problem'


    def set_loss_function(self, loss_function):
        """
        Sets loss function.

        :param loss_function: Loss function (e.g. nn.CrossEntropyLoss()) that will be set as the optimization criterion.
        """
        self.loss_function = loss_function

    def set_collate_function(self, collate_function):
        """ Sets collate function.

        :param collate_function: function that will be used to merge a list of individuals samples into a batch.
        """
        self.collate_fn = collate_function

    def __len__(self):
        """
        :return: The size of the dataset.

        """
        return self.length

    @abstractmethod
    def __getitem__(self, item):
        """
        Getter that returns an individual sample from the problem's associated dataset (that can be generated \
        on-the-fly, or retrieved from disk).

        To be redefined in subclasses.

        :param item: index of the sample to return.

        :return: DataDict containing the sample.

        """

    @abstractmethod
    def collate_fn(self, batch):
        """
        Generates a batch of samples from a list of individuals samples retrieved by __getitem__.
        The default collate_fn is torch.utils.data.default_collate.

        Abstract - to be defined in derived classes.

        :param batch: iterator (tensors, numbers, dicts or lists) of samples to merge into one batch.

        :return DataDict containing the created batch.

        """

    def evaluate_loss(self, data_dict, logits, _):
        """
        Calculates loss between the predictions/logits and targets (from data_dict) using the selected loss function.

        :param data_dict: DataDict containing inputs and targets.
        :param logits: Logits being output of the model.
        :param _: auxiliary tuple (aux_tuple) is not used in this function.
        """

        # Compute loss using the provided loss function. 
        loss = self.loss_function(logits, data_dict['targets'])

        return loss

    def add_statistics(self, stat_col):
        """
        Add statistics to collector.

        EMPTY - To be redefined in inheriting classes.

        :param stat_col: Statistics collector.

        """
        pass
        
    def collect_statistics(self, stat_col, data_dict, logits, _):
        """
        Base statistics collection.

        EMPTY - To be redefined in inheriting classes.

        :param stat_col: Statistics collector.
        :param data_dict: DataDict containing inputs and targets.
        :param logits: Logits being output of the model.
        :param _: auxiliary tuple (aux_tuple) is not used in this function.

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

        Used to reset statistics collector over one epoch, e.g.:
            - Average accuracy over the epoch
            - Time taken for the epoch and average per batch
            - etc...

        EMPTY - To be redefined in inheriting classes.

        """
        pass

    def finalize_epoch(self):
        """
        Function called at the end of an epoch to execute a few tasks, e.g.:
            - Compute the mean accuracy over the epoch
            - Get the time taken for the epoch and per batch
            - etc...

        EMPTY - To be redefined in inheriting classes.

        TODO: To display the final results for the current epoch, this function should use the Logger.
        """
        pass

    def plot_preprocessing(self, data_dict, logits):
        """
        Allows for some data preprocessing before the model creates a plot for visualization during training or
        inference.
        To be redefined in inheriting classes.
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

    data_definition = {'inputs': {'size': [64, 20], 'type': int}, 'targets': {'size': [64], 'type': int}}

    datadict = DataDict({key: None for key in data_definition.keys()})

    #datadict['inputs'] = torch.ones([64, 20, 512]).type(torch.FloatTensor)
    #datadict['targets'] = torch.ones([64, 20]).type(torch.FloatTensor)

    print(repr(datadict))
