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

    def __init__(self, dim, max_step, dtype):
        """
        Constructor for the QuestionDrivenController.

        :param dim: dimension of feature vectors
        :type dim: int

        :param max_step: maximum number of steps -> number of VWM cells in the network.
        :type max_step: int

        :param dtype

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
        self.temporal_classifier = torch.nn.Sequential(linear(2*dim, 2*dim, bias=True),
                                                       torch.nn.ReLU(),
                                                       linear(2*dim, 4, bias=True))

        self.sharpen = torch.nn.Parameter(torch.ones(1,).type(dtype))

    def forward(self, step, contextual_words, question_encoding, control_state, eps_tensor):
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

        :param eps_tensor

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
        new_cqi = torch.cat([new_control_state, pos_aware_question_encoding], dim=-1)
        tcw = self.temporal_classifier(new_cqi)

        tcw_smax = torch.softmax(tcw, dim=-1)

        sharpen = 1 + torch.nn.functional.softplus(self.sharpen)
        tcw_power = (tcw_smax + eps_tensor) ** sharpen

        tcw_sum = torch.max(torch.sum(tcw_power, dim=-1, keepdim=True), eps_tensor)

        temporal_class_weights = tcw_power / tcw_sum

        # # get t_now, t_last, t_latest, t_none from temporal_class_weights
        # t_now = temporal_class_weights[:, 0]
        # t_last = temporal_class_weights[:, 1]
        # t_latest = temporal_class_weights[:, 2]
        # t_none = temporal_class_weights[:, 3]
        # print(f'sharpness = {sharpen}')
        # print(t_last, t_latest, t_now, t_none)
        #
        #
        # return control and the temporal class weights
        return new_control_state, control_attention, temporal_class_weights
