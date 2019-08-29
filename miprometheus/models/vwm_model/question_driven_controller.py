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
question_driven_controller.py
"""
__author__ = "Vincent Albouy, T.S. Jayram"

import torch
from torch.nn import Module

from miprometheus.models.vwm_model.utils_VWM import linear
from miprometheus.models.vwm_model.attention_module import AttentionModule


class QuestionDrivenController(Module):
    """
    Implementation of the ``QuestionDrivenController`` 
    """

    def __init__(self, dim, max_step):
        """
        Constructor for the QuestionDrivenController.

        :param dim: dimension of feature vectors
        :type dim: int

        :param max_step: maximum number of steps -> number of VWM cells in the network.
        :type max_step: int

        """

        # call base constructor
        super(QuestionDrivenController, self).__init__()

        # define the linear layers (one per step) used to make the questions
        # encoding
        self.pos_aware_layers = torch.nn.ModuleList()
        for _ in range(max_step):
            self.pos_aware_layers.append(linear(2 * dim, dim, bias=True))

        # define the linear layer used to create the cqi values
        self.ctrl_question = linear(2 * dim, dim, bias=True)

        # instantiate attention module
        self.attention_module = AttentionModule(dim)

        # instantiate neural network for T (temporal classifier that outputs 4 classes)
        self.temporal_classifier = torch.nn.Sequential(linear(dim, dim, bias=True),
                                                       torch.nn.ReLU(),
                                                       linear(dim, 4, bias=True),
                                                       torch.nn.Softmax(dim=-1))

    def forward(self, step, contextual_words, question_encoding, control_state):
        """
        Forward pass of the ``QuestionDrivenController``.

        :param step: index of the current VWM cell.
        :type step: int

        :param contextual_words: containing the words encodings
        ('representation of each word in the context of the question')
        [batch_size x maxQuestionLength x dim]

        :param question_encoding: global representation of sentence
        [batch_size x (2*dim)]

        :param control_state: previous control state [batch_size x dim]

        :return: new_control_state: [batch_size x dim]
        :return: temporal_class_weights: soft classification representing \
        temporal context now/last/latest/none of current step [batch_size x 4]

        """

        # select current 'position aware' linear layer & pass questions through it
        pos_aware_question_encoding = self.pos_aware_layers[step](
            question_encoding)

        # concat control state and position aware question encoding
        cqi = torch.cat([control_state, pos_aware_question_encoding], dim=-1)

        # project from 2dim to 1dim
        cqi = self.ctrl_question(cqi)  # [batch_size x dim]

        # retrieve content new_control_state + attention control_attention
        new_control_state, control_attention = self.attention_module(cqi, contextual_words)

        # neural network  that returns temporal class weights
        temporal_class_weights = self.temporal_classifier(new_control_state)

        # return control and the temporal class weights
        return new_control_state, control_attention, temporal_class_weights
