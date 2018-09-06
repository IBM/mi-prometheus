#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""problem.py: contains base class for all problems"""
__author__      = "Tomasz Kornuta"


import collections
from abc import abstractmethod
from torch.utils.data import Dataset
import torch

_DataTuple = collections.namedtuple('DataTuple', ('inputs', 'targets'))
class DataTuple(_DataTuple):
    """Tuple used by storing batches of data by problems"""
    __slots__ = ()


_MaskAuxTuple = collections.namedtuple('MaskAuxTuple', ('mask'))
class MaskAuxTuple(_MaskAuxTuple):
    """
    Tuple used by storing batches of data by sequential problems using mask.
    Contains one element: mask that might be used for evaluation of the loss function.
    """
    __slots__ = ()


_LabelAuxTuple = collections.namedtuple('LabelAuxTuple', ('label'))
class LabelAuxTuple(_LabelAuxTuple):
    """
    Tuple used by storing batches of labels in classification problems.
    """
    __slots__ = ()


class Problem(Dataset):
    """ Class representing base class for all Problems.
    """

    def __init__(self, params):
        """ 
        Initializes problem object.

        :param params: Dictionary of parameters (read from configuration file).        
        """
        # Set default loss function.
        self.loss_function = None

        # Set default collate function
        self.collate_fn = torch.utils.data.default_collate

        # Store pointer to params.
        self.params = params

        # size of the dataset
        self.length = None

    def set_loss_function(self, loss_function):
        """ Sets loss function.

        :param loss_function: Loss function (e.g. nn.CrossEntropyLoss()) that will be set as optimization criterion.
        """
        self.loss_function = loss_function

    def set_collate_function(self, collate_function):
        """ Sets collate function.

        :param collate_function: Loss function that will be used to merge a list of individuals samples into a batch.
        """
        self.collate_function = collate_function

    @abstractmethod
    def __getitem__(self, item):
        """
        Getter that returns an invidual sample from the problem's associated dataset (that can be generated on-the-fly,
        or retrieved from disk).

        To be redefined in derived classes.
        :param item: index of the sample to return
        :return: sample.
        """

    def __len__(self):
        """
        :return: The size of the dataset.

        """
        return self.length

    @abstractmethod
    def collate_fn(self, batch):
        """
        Generates a batch of samples from a list of individuals samples retrieved by __getitem__.
        The default collate_fn is torch.utils.data.default_collate.

        Abstract - to be defined in derived classes.
        :param batch: iterator of samples to merge into one batch.

        ;:return created batch. #TODO: Should it be a DataTuple ?
        """

    def evaluate_loss(self, data_tuple, logits, _):
        """ Calculates loss between the predictions/logits and targets (from data_tuple) using the selected loss function.
        
        :param logits: Logits being output of the model.
        :param data_tuple: Data tuple containing inputs and targets.
        :param _: auxiliary tuple (aux_tuple) is not used in this function. 
        """
        # Unpack tuple.
        (_, targets) = data_tuple

        # Compute loss using the provided loss function. 
        loss = self.loss_function(logits, targets)

        return loss

    def add_statistics(self, stat_col):
        """
        Add statistics to collector. 
        EMPTY - To be redefined in inheriting classes.

        :param stat_col: Statistics collector.
        """
        pass
        
    def collect_statistics(self, stat_col, data_tuple, logits, _):
        """
        Base statistics collection. 
        EMPTY - To be redefined in inheriting classes.

        :param stat_col: Statistics collector.
        :param data_tuple: Data tuple containing inputs and targets.
        :param logits: Logits being output of the model.
        :param _: auxiliary tuple (aux_tuple) is not used in this function. 
        """
        pass

    def get_epoch_size(self):
        """
        Compute the number of iterations ('episodes') to do given the size of the dataset and the batch size.
        Note: we are compting the last batch, even though it might be smaller than the other ones if the size of the
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
        """

        # TODO
        pass

    def finalize_epoch(self):
        """
        Function called at the end of an epoch to execute a few tasks, e.g.:
            - Compute the mean accuracy over the epoch
            - Get the time taken for the epoch and per batch
            - etc...

        It should pass those values to the Logger
        """
        pass



'''
    def turn_on_cuda(self, data_tuple, aux_tuple):
        """ Enables computations on GPU - copies the input and target matrices (from DataTuple) to GPU.
        This method has to be overwritten in derived class if one decides to copy other matrices as well.

        :param data_tuple: Data tuple.
        :param aux_tuple: Auxiliary tuple (WARNING: Values stored in that variable will remain in CPU)
        :returns: Pair of Data and Auxiliary tuples (Data on GPU, Aux on CPU).
        """
        # Unpack tuples and copy data to GPU.
        gpu_inputs = data_tuple.inputs.cuda()
        gpu_targets = data_tuple.targets.cuda()

        # Pack matrices to tuples.
        data_tuple = DataTuple(gpu_inputs, gpu_targets)

        return data_tuple, aux_tuple

    def plot_preprocessing(self, data_tuple, aux_tuple, logits):
        """
        Allows for some data preprocessing before the model creates a plot for visualization during training or
        inference.
        To be redefined in inheriting classes.
        :param data_tuple: Data tuple.
        :param aux_tuple: Auxiliary tuple.
        :param logits: Logits being output of the model.
        :return: data_tuplem aux_tuple, logits after preprocessing.
        """
        return data_tuple, aux_tuple, logits


    def curriculum_learning_initialize(self, curriculum_params):
        """ 
        Initializes curriculum learning - simply saves the curriculum params.
        This method can be overwriten in the derived classes.

        :curriculum_params: Interface to parameters accessing curriculum learning view of the registry tree. 
        """
        # Save params.
        self.curriculum_params = curriculum_params


    def curriculum_learning_update_params(self, episode):
        """
        Updates problem parameters according to curriculum learning.
        There is no general solution to curriculum learning.
        This method should be overwriten in the derived classes.

        :param episode: Number of the current episode.
        :returns: True informing that CL wasn't active at all (i.e. is finished).
        """
        return True
'''