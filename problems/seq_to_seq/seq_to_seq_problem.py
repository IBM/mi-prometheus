#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""problem.py: contains base class for all seq2seq problems"""
__author__ = "Tomasz Kornuta"

from problems.problem import Problem, DataDict
import torch


class SeqToSeqProblem(Problem):
    """
    Class representing base class for all sequential problems.
    """

    def __init__(self, params):
        """
        Initializes problem object. Calls base constructor.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).

        """
        super(SeqToSeqProblem, self).__init__(params)

        # "Default" problem name.
        self.name = 'SeqToSeqProblem'

        # set default data_definitions dict
        self.data_definitions = {'sequences': {'size': [-1, -1, -1], 'type': [torch.Tensor]},
                                 'sequences_length': {'size': [-1, 1], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1, -1, -1], 'type': [torch.Tensor]},
                                 'mask': {'size': [-1, 1], 'type': [torch.Tensor]}
                                 }

        # Check if predictions/targets should be masked.
        if 'use_mask' not in params:
            # TODO: Set false as default!
            params.add_default_params({'use_mask': True})
        self.use_mask = params["use_mask"]

    def evaluate_loss(self, data_dict, logits):
        """ Calculates accuracy equal to mean number of correct predictions in a given batch.
        WARNING: Applies mask to both logits and targets!

        :param data_dict: DataDict({'sequences', 'sequences_length', 'targets', 'mask'}).

        :param logits: Predictions being output of the model.

        """
        # Check if mask should be is used - if so, use the correct loss
        # function.
        if self.use_mask:
            loss = self.loss_function(
                logits, data_dict['targets'], data_dict['mask'])
        else:
            loss = self.loss_function(logits, data_dict['targets'])

        return loss

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

        :param index: index of the sample to return.

        :return: DataDict containing the sample.

        """

        return DataDict({key: None for key in self.data_definitions.keys()})


if __name__ == '__main__':

    from utils.param_interface import ParamInterface

    sample = SeqToSeqProblem(ParamInterface())[0]
    # equivalent to ImageTextToClassProblem(params={}).__getitem__(index=0)

    print(repr(sample))
