#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2018 Kim Seonghyeon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ------------------------------------------------------------------------------
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
utils_mac.py: Implementation of utils methods for the MAC network. Cf https://arxiv.org/abs/1803.03067 for the \
reference paper.
"""
__author__ = "Vincent Albouy, T.S. Jayram"

from torch import nn

def linear(input_dim, output_dim, bias=True):

    """
    Defines a Linear layer. Specifies Xavier as the initialization type of the weights, to respect the original \
    implementation: https://github.com/stanfordnlp/mac-network/blob/master/ops.py#L20

    :param input_dim: input dimension
    :type input_dim: int

    :param output_dim: output dimension
    :type output_dim: int

    :param bias:  If set to True, the layer will learn an additive bias initially set to true \
    (as original implementation https://github.com/stanfordnlp/mac-network/blob/master/ops.py#L40)
    :type bias: bool

    :return: Initialized Linear layer
    :type: torch layer

    """

    #define linear layer from torch.nn library
    linear_layer = nn.Linear(input_dim, output_dim, bias=bias)

    #initialize weights
    nn.init.xavier_uniform_(linear_layer.weight)

    #initialize biases
    if bias:
        linear_layer.bias.data.zero_()

    return linear_layer


def eval_predicate(temporal_class_weights, valid_vo, valid_mo):
    # get t1,t2,t3,t4 from temporal_class_weights
    # corresponds to now, last, latest, or none
    t1 = temporal_class_weights[:, 0]
    t2 = temporal_class_weights[:, 1]
    t3 = temporal_class_weights[:, 2]
    t4 = temporal_class_weights[:, 3]

    # if the temporal context last or latest,
    # then do we replace the existing memory object?
    do_replace = valid_mo * valid_vo * (t2 + t3) * (1 - t4)

    # otherwise do we add a new one to memory?
    do_add_new = (1 - valid_mo) * valid_vo * (t2 + t3) * (1 - t4)

    # final read vector
    # (now or latest) and valid visual object?
    is_visual = (t1 + t3) * valid_vo
    # optional extra check that it is neither last nor none
    # is_visual = is_visual * (1 - t2) * (1 - t4)

    # (now or (latest and (not valid visual object))) and valid memory object?
    is_mem = (t2 + t3 * (1 - valid_vo)) * valid_mo
    # optional extra check that it is neither now nor none
    # is_mem = is_mem * (1 - t1) * (1 - t4)

    return do_replace, do_add_new, is_visual, is_mem


