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
thought_unit.py: Implementation of the ``ThoughtUnit`` for the VWM network.
"""
__author__ = "Vincent Albouy, T.S. Jayram"

import torch
from torch.nn import Module
from miprometheus.models.VWM_model.utils_VWM import linear

class ThoughtUnit(Module):

    """
    Implementation of the ``ThoughtUnit`` of the MAC network.
    """

    def __init__(self, dim):
        """
        Constructor for the ``ThoughtUnit``.

        :param dim: global 'd' hidden dimension
        :type dim: int

        """

        # call base constructor
        super(ThoughtUnit, self).__init__()

        # linear layer for the concatenation of context_output and summary_output
        self.concat_layer = linear(2 * dim, dim, bias=True)

    def forward(self, relevant_object, summary_object):
        """
        Forward pass of the ``SummaryUpdateUnit``.

        # :param is_visual
        # :param visual_object
        # :param is_mem
        # :param memory_object
        :param relevant_object
        :param summary_object

        :return: new_summary_object
        """

        # # combine the new context_output with the summary_output along dimension 1
        # next_context_output = self.concat_layer(torch.cat([context_output, summary_output], 1))
        #
        #
        # return next_context_output

        # compute new relevant object
        # relevant_object = (is_visual[..., None] * visual_object
        #                    + is_mem[..., None] * memory_object)

        # combine the new read vector with the prior memory state (w1)
        new_summary_object = self.concat_layer(
            torch.cat([relevant_object, summary_object], dim=1))

        return new_summary_object
