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
s_mac.py:

    - Implementation of the simplified ``MAC`` network (abbreviated as S-MAC), \
    reusing the different units implemented in separated files.
    - Cf https://arxiv.org/abs/1803.03067 for the reference MAC paper (Hudson and Manning, CVPR 2018).
    - Submitted to the 2018 NIPS VIGIL workshop. More information will be provided after reviews.


"""
__author__ = "Vincent Marois & T.S. Jayram"

import os
import nltk
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from miprometheus.models.model import Model

from miprometheus.models.mac.input_unit import InputUnit
from miprometheus.models.s_mac.s_mac_unit import MACUnit
from miprometheus.models.mac.output_unit import OutputUnit


class sMacNetwork(Model):
    """
    Implementation of the entire ``S-MAC`` model.

    .. note::

        This implementation is a simplified version of the MAC network, where modifications regarding the different \
        units have been done to reduce the number of linear layers (and thus number of parameters).

        This is part of a submission to the VIGIL workshop for NIPS 2018. More information (e.g. associated paper) \
        will be provided in the near future.


    """

    def __init__(self, params, problem_default_values_={}):
        """
        Constructor for the ``S-MAC`` network.

        :param params: dict of parameters (read from configuration ``.yaml`` file).
        :type params: :py:class:`miprometheus.utils.ParamInterface`

        :param problem_default_values_: default values coming from the :py:class:`Problem` class.
        :type problem_default_values_: dict
        """

        # call base constructor
        super(sMacNetwork, self).__init__(params, problem_default_values_)

        # parse params dict
        self.dim = params['dim']
        self.embed_hidden = params['embed_hidden']  # embedding dimension
        self.max_step = params['max_step']
        self.dropout = params['dropout']

        try:
            self.nb_classes = problem_default_values_['nb_classes']
        except Exception as ex:
            self.logger.warning("Couldn't retrieve one or more value(s) from problem_default_values_.")
            self.logger.warning("Exception: {}".format(ex))

        self.name = 'S-MAC'

        # instantiate units
        self.input_unit = InputUnit(dim=self.dim, embedded_dim=self.embed_hidden)

        self.mac_unit = MACUnit(dim=self.dim, max_step=self.max_step,
                                dropout=self.dropout)

        self.output_unit = OutputUnit(dim=self.dim, nb_classes=self.nb_classes)

        self.data_definitions = {'images': {'size': [-1, 1024, 14, 14], 'type': [np.ndarray]},
                                 'questions': {'size': [-1, -1, -1], 'type': [torch.Tensor]},
                                 'questions_length': {'size': [-1], 'type': [list, int]},
                                 'targets': {'size': [-1, self.nb_classes], 'type': [torch.Tensor]}
                                 }

        # transform for the image plotting
        self.transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])

    def forward(self, data_dict, dropout=0.15):
        """
        Forward pass of the ``S-MAC`` network.

        Calls first the :py:class:`InputUnit`, then the recurrent  S-MAC cells and finally the :py:class:`OutputUnit``.

        :param data_dict: input data batch.
        :type data_dict: :py:class:`miprometheus.utils.DataDict`

        :param dropout: dropout rate.
        :type dropout: float

        :return: Predictions of the model.
        """

        # reset cell state history for visualization
        if self.app_state.visualize:
            self.mac_unit.cell_state_history = []

        # unpack data_dict
        images = data_dict['images']
        questions = data_dict['questions']
        questions_length = data_dict['questions_length']

        # input unit: Ignore knowledge_base (feature maps) as not used at all.
        _, kb_proj, lstm_out, h = self.input_unit(questions, questions_length, images)

        # recurrent S-MAC cells
        memory = self.mac_unit(lstm_out, h, kb_proj)

        # output unit
        logits = self.output_unit(memory, h)

        return logits

    @staticmethod
    def generate_figure_layout():
        """
        Generate a figure layout for the attention visualization (done in :py:func:`sMacNetwork.plot()`)

        :return: :py:class:`matplotlib.figure.Figure` layout.

        """
        import matplotlib

        params = {'axes.titlesize': 'large',
                  'axes.labelsize': 'large',
                  'xtick.labelsize': 'medium',
                  'ytick.labelsize': 'medium'}
        matplotlib.pylab.rcParams.update(params)

        # Prepare "generic figure template".
        # Create figure object.
        fig = matplotlib.figure.Figure()

        # Create a specific grid for S-MAC.
        gs = matplotlib.gridspec.GridSpec(6, 2)

        # subplots: original image, attention on image & question, step index
        ax_image = fig.add_subplot(gs[2:6, 0])
        ax_attention_image = fig.add_subplot(gs[2:6, 1])
        ax_attention_question = fig.add_subplot(gs[0, :])
        ax_step = fig.add_subplot(gs[1, 0])

        # Set axis ticks
        ax_image.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_image.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_attention_image.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_attention_image.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        # question ticks
        ax_attention_question.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=40))

        ax_step.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_step.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        fig.set_tight_layout(True)

        return fig

    def plot(self, data_dict, logits, sample=0):
        """
        Visualize the attention weights (:py:class:`ControlUnit` & :py:class:`ReadUnit`) on the \
        question & feature maps.

        Dynamic visualization throughout the reasoning steps is possible.

        :param data_dict: DataDict({'questions_string', 'questions_type', 'targets_string','imgfiles', \
        'prediction_string', 'clevr_dir', **})
        :type data_dict: :py:class:`miprometheus.utils.DataDict`

        :param logits: Prediction of the model.
        :type logits: :py:class:`torch.Tensor`

        :param sample: Index of sample in batch (Default: 0)
        :type sample: int

        :return: True when the user closes the window, False if we do not need to visualize.

        """

        # check whether the visualization is required
        if not self.app_state.visualize:
            return False

        # Initialize timePlot window - if required.
        if self.plotWindow is None:
            from miprometheus.utils.time_plot import TimePlot
            self.plotWindow = TimePlot()

        # unpack data_dict
        s_questions = data_dict['questions_string']
        question_type = data_dict['questions_type']
        answer_string = data_dict['targets_string']
        imgfiles = data_dict['imgfiles']
        prediction_string = data_dict['predictions_string']
        clevr_dir = data_dict['clevr_dir']

        # needed for nltk.word.tokenize
        nltk.download('punkt')
        # tokenize question string using same processing as in the problem class
        words = nltk.word_tokenize(s_questions[sample])

        # Create figure template.
        fig = self.generate_figure_layout()
        # Get axes that artists will draw on.
        (ax_image, ax_attention_image, ax_attention_question, ax_step) = fig.axes

        # get the original image
        set = imgfiles[sample].split('_')[1]
        image = os.path.join(clevr_dir, 'images', set, imgfiles[sample])
        image = Image.open(image).convert('RGB')
        image = self.transform(image)
        image = image.permute(1, 2, 0)  # [300, 300, 3]

        # get most probable answer -> prediction of the network
        proba_answers = torch.nn.functional.softmax(logits, -1)
        proba_answer = proba_answers[sample].detach().cpu()
        proba_answer = proba_answer.max().numpy()

        # image & attention sizes
        width = image.size(0)
        height = image.size(1)

        frames = []
        for step, (attention_mask, attention_question) in zip(
                range(self.max_step), self.mac_unit.cell_state_history):
            # preprocess attention image, reshape
            attention_size = int(np.sqrt(attention_mask.size(-1)))
            # attention mask has size [batch_size x 1 x(H*W)]
            attention_mask = attention_mask.view(-1, 1, attention_size, attention_size)

            # upsample attention mask
            m = torch.nn.Upsample(size=[width, height], mode='bilinear', align_corners=True)
            up_sample_attention_mask = m(attention_mask)
            attention_mask = up_sample_attention_mask[sample, 0]

            # preprocess question, pick one sample number
            attention_question = attention_question[sample]

            # Create "Artists" drawing data on "ImageAxes".
            num_artists = len(fig.axes) + 1
            artists = [None] * num_artists

            # set title labels
            ax_image.set_title('CLEVR image: {}'.format(imgfiles[sample]))
            ax_attention_question.set_xticklabels(['h'] + words, rotation='vertical', fontsize=10)
            ax_step.axis('off')

            # set axis attention labels
            ax_attention_image.set_title('Predicted Answer: ' + prediction_string[sample] +
                                         ' [ proba: ' + str.format("{0:.3f}", proba_answer) + ']  ' +
                                         'Ground Truth: ' + answer_string[sample])

            # Tell artists what to do:
            artists[0] = ax_image.imshow(image, interpolation='nearest', aspect='auto')
            artists[1] = ax_attention_image.imshow(image, interpolation='nearest', aspect='auto')
            artists[2] = ax_attention_image.imshow(attention_mask,
                                                   interpolation='nearest',
                                                   aspect='auto',
                                                   alpha=0.5,
                                                   cmap='Reds')
            artists[3] = ax_attention_question.imshow(attention_question.transpose(1, 0),
                                                      interpolation='nearest', aspect='auto', cmap='Reds')
            artists[4] = ax_step.text(0, 0.5, 'Reasoning step index: ' + str(step) +
                                      ' | Question type: ' + question_type[sample], fontsize=15)

            # Add "frame".
            frames.append(artists)

        # Plot figure and list of frames.
        self.plotWindow.update(fig, frames)

        return self.plotWindow.is_closed


if __name__ == '__main__':
    dim = 512
    embed_hidden = 300
    max_step = 12
    self_attention = True
    memory_gate = True
    nb_classes = 28
    dropout = 0.15

    from miprometheus.utils.app_state import AppState
    from miprometheus.utils.param_interface import ParamInterface
    from torch.utils.data.dataloader import DataLoader
    app_state = AppState()

    from miprometheus.problems.image_text_to_class.clevr import CLEVR
    problem_params = ParamInterface()
    problem_params.add_config_params({'settings': {'data_folder': '~/Downloads/CLEVR_v1.0',
                                                   'set': 'train', 'dataset_variant': 'CLEVR'},

                                      'images': {'raw_images': False,
                                                 'feature_extractor': {'cnn_model': 'resnet101',
                                                                       'num_blocks': 4}},

                                      'questions': {'embedding_type': 'random', 'embedding_dim': 300}})

    # create problem
    clevr_dataset = CLEVR(problem_params)
    print('Problem {} instantiated.'.format(clevr_dataset.name))

    # instantiate DataLoader object
    batch_size = 64
    problem = DataLoader(clevr_dataset, batch_size=batch_size, collate_fn=clevr_dataset.collate_fn)

    model_params = ParamInterface()
    model_params.add_config_params({'dim': dim,
                                    'embed_hidden': embed_hidden,
                                    'max_step': 12,
                                    'dropout': dropout})

    model = sMacNetwork(model_params, clevr_dataset.default_values)
    print('Model {} instantiated.'.format(model.name))
    model.app_state.visualize = True

    # perform handshaking between MAC & CLEVR
    model.handshake_definitions(clevr_dataset.data_definitions)

    # generate a batch
    for i_batch, sample in enumerate(problem):
        print('Sample # {} - {}'.format(i_batch, sample['images'].shape), type(sample))
        logits = model(sample)
        clevr_dataset.plot_preprocessing(sample, logits)
        model.plot(sample, logits)
        print(logits.shape)

    print('Unit test completed.')
