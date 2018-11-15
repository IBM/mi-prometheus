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
relational_network.py: contains the implementation of the Relational Network model from DeepMind.
See the reference paper here: https://arxiv.org/pdf/1706.01427.pdf.
"""
__author__ = "Vincent Marois"

import torch

from miprometheus.models.model import Model

from miprometheus.models.relational_net.conv_input_model import ConvInputModel
from miprometheus.models.relational_net.functions import PairwiseRelationNetwork, SumOfPairsAnalysisNetwork


class RelationalNetwork(Model):
    """
    Implementation of the Relational Network (RN) model.

    Questions are processed with an LSTM to produce a question embedding, and images are processed \
    with a CNN to produce a set of objects for the RN. 'Objects' are constructed using feature-map vectors \
    from the convolved image.  The RN considers relations across all pairs of objects, conditioned on the
    question embedding, and integrates all these relations to answer the question.

    Reference paper: https://arxiv.org/abs/1706.01427.

    The CNN model used for the image encoding is located in ``conv_input_model.py``.

    The MLPs (g_theta & f_phi) are in ``functions.py``.

    .. warning:

        This implementation has only been tested on the ``SortOfCLEVR`` problem class proposed in this \
        framework and will require modification to work on the CLEVR dataset (also proposed in this framework).\

        This should be addressed in a future release.


    """

    def __init__(self, params, problem_default_values_={}):
        """
        Constructor.

        Instantiates the CNN model (4 layers), and the 2 Multi Layer Perceptrons.

        :param params: dictionary of parameters (read from the ``.yaml`` configuration file.)

        :param problem_default_values_: default values coming from the ``Problem`` class.
        :type problem_default_values_: dict.

        """

        # call base constructor
        super(RelationalNetwork, self).__init__(params, problem_default_values_)

        self.name = 'RelationalNetwork'

        # instantiate conv input model for image encoding
        self.cnn_model = ConvInputModel()

        try:
            # get image information from the problem class
            self.num_channels = problem_default_values_['num_channels']  # number of channels
            self.height = problem_default_values_['height']
            self.width = problem_default_values_['width']
            self.question_size = problem_default_values_['question_size']

            # number of output nodes
            self.nb_classes = problem_default_values_['num_classes']

        except KeyError:
            self.logger.warning("Couldn't retrieve one or more value(s) from problem_default_values_.")

        # compute the length of the input to the g_theta MLP:
        input_size = ( self.cnn_model.conv4.out_channels + 2) *2 + self.question_size
        # instantiate network to compare regions pairwise
        self.pair_network = PairwiseRelationNetwork(input_size=input_size)

        # instantiate network to analyse the sum of the pairs
        self.sum_network = SumOfPairsAnalysisNetwork(output_size=self.nb_classes)

        self.data_definitions = {'images': {'size': [-1, self.num_channels, self.height, self.width],
                                            'type': [torch.Tensor]},
                                 'questions': {'size': [-1, -1, -1], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1, 1], 'type': [torch.Tensor]}
                                 }

    def build_coord_tensor(self, batch_size, d):
        """
        Create the tensor containing the spatial relative coordinate of each \
        region (1 pixel) in the feature maps of the ``ConvInputModel``. These \
        spatial relative coordinates are used to 'tag' the regions.

        :param batch_size: batch size
        :type batch_size: int

        :param d: size of 1 feature map
        :type d: int

        :return: tensor of shape [batch_size x d x d x 2]

        """
        coords = torch.linspace(-1 / 2., 1 / 2., d)
        x = coords.unsqueeze(0).repeat(d, 1)
        y = coords.unsqueeze(1).repeat(1, d)
        ct = torch.stack((x, y)).type(self.app_state.dtype)  # [2 x d x d]

        # broadcast to all batches
        # [batch_size x 2 x d x d]
        ct = ct.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # indicate that we do not track gradient for this tensor
        ct.requires_grad = False

        return ct

    def forward(self, data_dict):
        """
        Runs the ``RelationalNetwork`` model.

        :param data_dict: DataDict({'images', 'questions', ...}) containing:

            - images [batch_size, num_channels, height, width],
            - questions [batch_size, question_size]


        :type data_dict: utils.DataDict

        :returns: Predictions of the model [batch_size, nb_classes]

        """

        images = data_dict['images'].type(self.app_state.dtype)
        questions = data_dict['questions']

        question_size = questions.shape[-1]

        # step 1 : encode images
        feature_maps = self.cnn_model(images)
        batch_size = feature_maps.shape[0]
        # number of kernels in the final convolutional layer
        k = feature_maps.shape[1]
        d = feature_maps.shape[2]  # size of 1 feature map

        # step 2: 'tag' all regions in feature_maps with their relative spatial
        # coordinates
        ct = self.build_coord_tensor(batch_size, d)  # [batch_size x 2 x d x d]
        x_ct = torch.cat([feature_maps, ct], 1)  # [batch_size x (k+2) x d x d]
        # update number of channels
        k += 2

        # step 3: form all possible pairs of region in feature_maps (d** 2 regions -> d ** 4 pairs!)
        # flatten out feature_maps: [batch_size x k x d x d] -> [batch_size x k
        # x (d ** 2)]
        x_ct = x_ct.view(batch_size, k, d**2)
        x_ct = x_ct.transpose(2, 1)  # [batch_size x (d ** 2) x k]

        x_i = x_ct.unsqueeze(1)  # [batch_size x 1 x (d ** 2) x k]
        # [batch_size x (d ** 2) x (d ** 2) x k]
        x_i = x_i.repeat(1, (d**2), 1, 1)

        # step 4: add the question everywhere
        questions = questions.unsqueeze(1).repeat(
            1, d ** 2, 1)  # [batch_size, (d**2), question_size]
        # [batch_size, (d**2), 1, question_size]
        questions = questions.unsqueeze(2)

        x_j = x_ct.unsqueeze(2)  # [batch_size x (d ** 2) x 1 x k]
        # [batch_size x (d ** 2) x 1 x (k+qst_size)]
        x_j = torch.cat([x_j, questions], dim=-1)
        # [batch_size x (d ** 2) x (d ** 2) x (k+qst_size)]
        x_j = x_j.repeat(1, 1, (d**2), 1)

        # generate all pairs
        # [batch_size, (d**2), (d**2), 2*k+qst_size]
        x = torch.cat([x_i, x_j], dim=-1)

        # step 5: pass pairs through pair_network
        # reshape for passing through network
        input_size = 2 * k + question_size
        x = x.view(batch_size * (d ** 4), input_size)
        x_g = self.pair_network(x)

        # reshape again & element-wise sum on the second dimension
        x_g = x_g.view(batch_size, (d ** 4), 256)
        x_f = x_g.sum(1)

        # step 6: pass sum of pairs through sum_network
        x_out = self.sum_network(x_f)

        return x_out


if __name__ == '__main__':
    """Unit test for the RelationalNetwork on SortOfCLEVR"""
    from miprometheus.utils.app_state import AppState
    from miprometheus.utils.param_interface import ParamInterface
    from torch.utils.data import DataLoader
    app_state = AppState()

    from miprometheus.problems.image_text_to_class.sort_of_clevr import SortOfCLEVR
    problem_params = ParamInterface()
    problem_params.add_config_params({'data_folder': '~/data/sort-of-clevr/',
                                      'split': 'train',
                                      'regenerate': False,
                                      'dataset_size': 10000,
                                      'img_size': 128})

    # create problem
    sort_of_clevr = SortOfCLEVR(problem_params)
    print('Problem {} instantiated.'.format(sort_of_clevr.name))

    # instantiate DataLoader object
    batch_size = 64
    problem = DataLoader(sort_of_clevr, batch_size=batch_size, collate_fn=sort_of_clevr.collate_fn)

    model_params = ParamInterface()
    model_params.add_config_params({})

    model = RelationalNetwork(model_params, sort_of_clevr.default_values)
    print('Model {} instantiated.'.format(model.name))
    model.app_state.visualize = True

    # perform handshaking between RN & SortOfCLEVR
    model.handshake_definitions(sort_of_clevr.data_definitions)

    # generate a batch
    for i_batch, sample in enumerate(problem):
        print('Sample # {} - {}'.format(i_batch, sample['images'].shape), type(sample))
        logits = model(sample)
        sort_of_clevr.plot_preprocessing(sample, logits)
        model.plot(sample, logits)
        print(logits.shape)

    print('Unit test completed.')
