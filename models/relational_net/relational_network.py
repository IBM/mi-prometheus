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

"""relational_network.py: contains the implementation of the Relational Network."""
__author__      = "Vincent Marois"

from models.relational_net.conv_input_model import ConvInputModel
from models.relational_net.functions import g_theta, f_phi

import torch
import numpy as np

from models.model import Model
from misc.app_state import AppState
app_state = AppState()


class RelationalNetwork(Model):
    """
    Implementation of the Relational Network model. Reference paper: https://arxiv.org/abs/1706.01427
    The CNN model used for the image encoding is located in .conv_input_model.py
    The MLPs (g_theta & f_phi) are in .functions.
    """

    def __init__(self, params):
        """
        Constructor
        :param params: dict of parameters.
        """

        # call base constructor
        super(RelationalNetwork, self).__init__(params)

        # instantiate conv input model for image encoding
        self.cnn_model = ConvInputModel()

        # instantiate g_theta function
        self.g_theta = g_theta(params['g_theta'])

        # instantiate f_phi function
        self.f_phi = f_phi(params['f_phi'])

        # TODO: Anything else??

    def build_coord_tensor(self, batch_size, d):
        """
        Create the tensor containing the spatial relative coordinate of each region (1 pixel) in the feature maps of the
        ConvInputModel.
        These spatial relative coordinates are used to 'tag' the regions.

        :param batch_size: batch size
        :param d: size of 1 feature map

        :return: tensor of shape [batch_size x d x d x 2]
        """
        coords = torch.linspace(-d / 2., d / 2., d)
        x = coords.unsqueeze(0).repeat(d, 1)
        y = coords.unsqueeze(1).repeat(1, d)
        ct = torch.stack((x, y))  # [2 x d x d]

        # broadcast to all batches
        ct = ct.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size x 2 x d x d]

        # indicate that we do not track gradient for this tensor
        ct.requires_grad = False

        return ct

    def forward(self, data_tuple):
        """
        Runs the RelationalNetwork model.

        :param data_tuple: Tuple containing images [batch_size, num_channels, height, width]
        and questions [batch_size, question_size]
        :returns: output [batch_size, nb_classes]
        """

        # unpack datatuple
        (images, questions), _ = data_tuple
        question_size = questions.shape[-1]

        # step 1 : encode images
        feature_maps = self.cnn_model(images)
        batch_size = feature_maps.shape[0]
        k = feature_maps.shape[1]  # number of kernels in the final convolutional layer
        d = feature_maps.shape[2]  # size of 1 feature map

        # step 2: 'tag' all regions in feature_maps with their relative spatial coordinates
        ct = self.build_coord_tensor(batch_size, d)  # [batch_size x 2 x d x d]
        x_ct = torch.cat([feature_maps, ct], 1)  # [batch_size x (k+2) x d x d]
        # update number of channels
        k += 2

        # step 3: form all possible pairs of region in feature_maps (d** 2 regions -> d ** 4 pairs!)
        # flatten out feature_maps: [batch_size x k x d x d] -> [batch_size x k x (d ** 2)]
        x_ct = x_ct.view(batch_size, k, d**2)
        x_ct = x_ct.transpose(2, 1)  # [batch_size x (d ** 2) x k]

        x_i = x_ct.unsqueeze(1)  # [batch_size x 1 x (d ** 2) x k]
        x_i = x_i.repeat(1, (d**2), 1, 1)  # [batch_size x (d ** 2) x (d ** 2) x k]

        x_j = x_ct.unsqueeze(2)  # [batch_size x (d ** 2) x 1 x k]
        x_j = x_j.repeat(1, 1, (d**2), 1)

        # concatenate all together
        x_pairs = torch.cat([x_i, x_j], -1)  # [batch_size, (d**2), (d**2), 2*k]

        # step 4: add the question everywhere
        questions = questions.unsqueeze(1).repeat(1, d**2, 1)  # [batch_size, (d**2), question_size]
        questions = questions.unsqueeze(2).repeat(1, 1, d**2, 1)  # [batch_size, (d**2), (d**2), question_size]

        # add the question encoding at index 0:
        x = torch.cat([questions, x_pairs], dim=-1)

        # step 5: pass through g_theta
        # reshape for passing through network
        input_size = 2*k + question_size
        x = x.view(batch_size, (d ** 4), input_size)
        x_g = self.g_theta(x)

        # element-wise sum on the second dimension
        x_f = x_g.sum(1)

        # step 6: pass through f_phi
        x_out = self.f_phi(x_f)

        return x_out


if __name__ == '__main__':

    question_size = 13
    input_size = (24 + 2) * 2 + question_size
    output_size = 29
    params = {'g_theta': {'input_size': input_size},
              'f_phi': {'output_size': output_size}}

    batch_size = 128
    img_size = 128
    images = np.random.binomial(1, 0.5, (batch_size, 3, img_size, img_size))
    images = torch.from_numpy(images).type(app_state.dtype)

    questions = np.random.binomial(1, 0.5, (batch_size, question_size))
    questions = torch.from_numpy(questions).type(app_state.dtype)

    targets = None

    net = RelationalNetwork(params)

    net(((images, questions), targets))