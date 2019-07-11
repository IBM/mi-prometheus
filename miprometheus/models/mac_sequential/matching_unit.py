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
read_unit.py: Implementation of the ``MatchingUnit`` for the VWM network. 

"""
__author__ = "Vincent Albouy"

import torch
from torch.nn import Module

from miprometheus.models.mac_sequential.utils_VWM import linear


class MatchingUnit(Module):
    """
    Implementation of the `` MatchingUnit`` of the VWM network.
    """

    def __init__(self, dim):
        """
        Constructor for the `` MatchingUnit``.
        :param dim: global 'd' hidden dimension
        :type dim: int
        """

        # call base constructor
        super(MatchingUnit, self).__init__()

        self.linear_read = torch.nn.Sequential(linear(2 * dim, 2 * dim, bias=True),
                                               torch.nn.ELU(),
                                               linear(2 * dim, 1, bias=True))

        self.linear_read_history = torch.nn.Sequential(linear(2 * dim, 2 * dim, bias=True),
                                                       torch.nn.ELU(),
                                                       linear(2 * dim, 1, bias=True))


    def forward(self, control, vo, mo):
        """
        Forward pass of the ``MatchingUnit``. Assuming 1 scalar attention weight per \
        knowledge base elements.
        :param control: last control state
        :type control: torch.tensor
        :param  vo: visual output
        :type vo: torch.tensor
        :param  mo: memory output
        :type mo: torch.tensor
   
        :return: gvt, gmt : visual gate and memory gate
        """
        # calculate gate gvt
        concat_read_visual = torch.cat([control, vo], dim=1)
        gvt = self.linear_read(concat_read_visual )
        gvt = torch.sigmoid(gvt)

        # calculate gate gmt
        concat_read_memory = torch.cat([control, mo], dim=1)
        gmt = self.linear_read_history(concat_read_memory)
        gmt = torch.sigmoid(gmt)

        #return two gates gvt, gmt
        return gvt, gmt