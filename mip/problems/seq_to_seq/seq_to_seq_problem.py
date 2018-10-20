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
seq_to_seq_problem.py: contains base class for all sequence to sequence problems.

"""

__author__ = "Tomasz Kornuta & Vincent Marois"

from mip.problems.problem import Problem
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

        # Set default data_definitions dict for all Seq2Seq problems.
        self.data_definitions = {'sequences': {'size': [-1, -1, -1], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1, -1, -1], 'type': [torch.Tensor]},
                                 'masks': {'size': [-1, -1, 1], 'type': [torch.Tensor]},
                                 'sequences_length': {'size': [-1, 1], 'type': [torch.Tensor]}
                                 }


        # Check if predictions/targets should be masked (DEFAULT: True).
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
                logits, data_dict['targets'], data_dict['masks'])
        else:
            loss = self.loss_function(logits, data_dict['targets'])

        return loss


if __name__ == '__main__':

    from mip.utils.param_interface import ParamInterface

    sample = SeqToSeqProblem(ParamInterface())[0]
    # equivalent to ImageTextToClassProblem(params={}).__getitem__(index=0)

    print(repr(sample))
