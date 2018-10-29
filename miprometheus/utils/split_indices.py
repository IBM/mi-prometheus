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
split_indices.py:

    - Contains the definition of a split_indices function.

"""
__author__ = "Tomasz Kornuta"

import numpy as np
from numpy.random import permutation


def split_indices(length, split, logger, random_sampling=True):
    """
    Function splits the indices of an array of a given ''length'' into two parts, using the ''split'' as divider.
    Random sampling is used by default, but can be turned off.

    :param length: Length (size) of the dataset.

    :param split: Determines how many indices will belong to subset a and supset b.

    :param logger: Logging utility.

    :param random_sampling: Use random sampling (DEFAULT: True). If set to false, will return two ranges instead of lists with indices.

    :returns: Two lists with indices (when random_sampling is True), or two lists bith two elements - ranges (when false).
    """  
    if random_sampling:
        logger.info('Using random sampling')
        # Random indices.
        indices = permutation(length)
        # Split into two pieces.
        split_a = indices[0:split]
        split_b = indices[split:length]
    else:
        logger.info('Splitting into two ranges without random sampling')
        # Split into two ranges.
        split_a = np.asarray([0,split-1], dtype=int)
        split_b = np.asarray([split,length-1], dtype=int)
    return split_a, split_b
