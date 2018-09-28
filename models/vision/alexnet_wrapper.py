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


import torchvision.models as models
import torch

from models.model import Model


class AlexnetWrapper(Model):
    """
    Wrapper class to Alexnet model from TorchVision.
    """

    def __init__(self, params):
        super(AlexnetWrapper, self).__init__(params)

        # set model from torchvision
        self.model = models.alexnet(params["num_classes"])

    def forward(self, data_tuple):

        # get data
        (x, _) = data_tuple

        # construct the three channels needed for alexnet
        if x.size(1) != 3:
            # inputs_size = (batch_size, num_channel, numb_columns, num_rows)
            num_channel = 3
            inputs_size = (x.size(0), num_channel, x.size(2), x.size(3))
            inputs = torch.zeros(inputs_size)

            for i in range(num_channel):
                inputs[:, None, i, :, :] = x
        else:
            inputs = x

        outputs = self.model(inputs)

        return outputs
