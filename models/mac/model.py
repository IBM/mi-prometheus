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

"""model.py: Implementation of the MAC network, reusing the different units implemented in separated files.
            Cf https://arxiv.org/abs/1803.03067 for the reference paper."""
__author__ = "Vincent Marois , Vincent Albouy"

# Add path to main project directory
import os
import torch.nn.functional as F

from models.model import Model
from misc.app_state import AppState

app_state = AppState()

from models.mac.input_unit import InputUnit
from models.mac.mac_unit import MACUnit
from models.mac.output_unit import OutputUnit
from PIL import Image
from torchvision import transforms

# visualization
import nltk
nltk.download('punkt')  # needed for nltk.word.tokenize


class MACNetwork(Model):
    """
    Implementation of the entire MAC network.
    """

    def __init__(self, params):
        """
        Constructor for the MAC network.
        :param params: dict of parameters.
        """

        # call base constructor
        super(MACNetwork, self).__init__(params)

        # parse params dict
        self.dim = params['dim']
        self.embed_hidden = params['embed_hidden']
        self.max_step = params['max_step']
        self.self_attention = params['self_attention']
        self.memory_gate = params['memory_gate']
        self.nb_classes = params['nb_classes']
        self.dropout = params['dropout']

        self.image = []

        # instantiate units
        self.input_unit = InputUnit(dim=self.dim, embedded_dim=self.embed_hidden)

        self.mac_unit = MACUnit(dim=self.dim, max_step=self.max_step, self_attention=self.self_attention,
                                memory_gate=self.memory_gate, dropout=self.dropout)

        self.output_unit = OutputUnit(dim=self.dim, nb_classes=self.nb_classes)

        # transform for the image plotting
        self.transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])

    def forward(self, data_tuple, dropout=0.15):

        # reset cell state history for visualization
        if self.app_state.visualize:
            self.mac_unit.cell_state_history = []

        # unpack data_tuple
        inner_tuple, _ = data_tuple
        image_questions_tuple, questions_len = inner_tuple
        images, questions = image_questions_tuple

        # input unit
        img, kb_proj, lstm_out, h = self.input_unit(questions, questions_len, images)
        self.image = kb_proj

        # recurrent MAC cells
        memory = self.mac_unit(lstm_out, h, img, kb_proj)

        # output unit
        logits = self.output_unit(memory, h)

        return logits

    def generate_figure_layout(self):
        """
        Generate a figure layout for the attention visualization (done in MACNetwork.plot())

        :return: figure layout.
        """

        from matplotlib.figure import Figure
        import matplotlib.ticker as ticker
        import matplotlib.gridspec as gridspec
        import matplotlib.pylab as pylab

        params = {'axes.titlesize': 'large',
                  'axes.labelsize': 'large',
                  'xtick.labelsize': 'medium',
                  'ytick.labelsize': 'medium'}
        pylab.rcParams.update(params)

        # Prepare "generic figure template".
        # Create figure object.
        fig = Figure()

        # Create a specific grid for MAC.
        gs = gridspec.GridSpec(6, 2)

        # subplots: original image, attention on image & question, step index
        ax_image = fig.add_subplot(gs[2:6, 0])
        ax_attention_image = fig.add_subplot(gs[2:6, 1])
        ax_attention_question = fig.add_subplot(gs[0, :])
        ax_step = fig.add_subplot(gs[1, 0])

        # Set axis ticks
        ax_image.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_image.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_attention_image.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_attention_image.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # question ticks
        ax_attention_question.xaxis.set_major_locator(ticker.MaxNLocator(nbins=40))

        ax_step.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_step.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        fig.set_tight_layout(True)

        return fig

    def plot(self, aux_tuple, logits, sample_number=0):
        """
        Visualize the attention weights (Control Unit & Read Unit) on the question & feature maps.
        Dynamic visualization trhoughout the reasoning steps possible.

        :param aux_tuple: aux_tuple (), transformed by CLEVR.plot_preprocessing()
            -> (s_questions, answer_string, imgfiles, set, prediction_string, clevr_dir)
        :param logits: prediction of the network
        :param sample_number: Number of sample in batch (DEFAULT: 0)

        :return: True when the user closes the window, False if we do not need to visualize.
        """

        # check whether the visualization is required
        if not self.app_state.visualize:
            return False

        # Initialize timePlot window - if required.
        if self.plotWindow is None:
            from misc.time_plot import TimePlot
            self.plotWindow = TimePlot()

        # attention mask [batch_size x 1 x(H*W)]

        # unpack aux_tuple
        (s_questions, answer_string, imgfiles, set, prediction_string, clevr_dir) = aux_tuple

        # tokenize question string using same processing as in the problem class
        words = nltk.word_tokenize(s_questions[sample_number])

        # Create figure template.
        fig = self.generate_figure_layout()
        # Get axes that artists will draw on.
        (ax_image, ax_attention_image, ax_attention_question, ax_step) = fig.axes

        # get the image
        image = os.path.join(clevr_dir, 'images', set, imgfiles[sample_number])
        image = Image.open(image).convert('RGB')
        image = self.transform(image)
        image = image.permute(1, 2, 0)  # [300, 300, 3]

        # get most probable answer -> prediction of the network
        proba_answers = F.softmax(logits, -1)
        proba_answer = proba_answers[sample_number].detach().cpu()
        proba_answer = proba_answer.max().numpy()

        # image & attention sizes
        width = image.size(0)
        height = image.size(1)

        frames = []
        for step, (attention_mask, attention_question) in zip(range(self.max_step), self.mac_unit.cell_state_history):
            # preprocess attention image, reshape
            attention_size = int(np.sqrt(attention_mask.size(-1)))
            attention_mask = attention_mask.view(-1, 1, attention_size, attention_size)

            # upsample attention mask
            m = torch.nn.Upsample(size=[width, height], mode='bilinear', align_corners=True)
            up_sample_attention_mask = m(attention_mask)
            attention_mask = up_sample_attention_mask[sample_number, 0]

            # preprocess question, pick one sample number
            attention_question = attention_question[sample_number]

            # Create "Artists" drawing data on "ImageAxes".
            num_artists = len(fig.axes) + 1
            artists = [None] * num_artists

            # set title labels
            ax_image.set_title('CLEVR image: {}'.format(imgfiles[sample_number]))
            ax_attention_question.set_xticklabels(['h'] + words, rotation='vertical', fontsize=10)
            ax_step.axis('off')

            # set axis attention labels
            ax_attention_image.set_title(
                'Predicted Answer: ' + prediction_string[sample_number] +
                ' [ proba: ' + str.format("{0:.3f}", proba_answer) + ']  ' + 'Ground Truth: ' +
                answer_string[sample_number])

            # Tell artists what to do:
            artists[0] = ax_image.imshow(image, interpolation='nearest', aspect='auto')
            artists[1] = ax_attention_image.imshow(image, interpolation='nearest', aspect='auto')
            artists[2] = ax_attention_image.imshow(attention_mask, interpolation='nearest', aspect='auto', alpha=0.5,
                                                   cmap='Reds')
            artists[3] = ax_attention_question.imshow(attention_question.transpose(1, 0), interpolation='nearest',
                                                      aspect='auto', cmap='Reds')
            artists[4] = ax_step.text(0, 0.5, 'Reasoning step index: ' + str(step), fontsize=15)

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

    from misc.param_interface import ParamInterface

    params = ParamInterface()
    params.add_custom_params(
        {'dim': dim, 'embed_hidden': embed_hidden, 'max_step': 12, 'self_attention': self_attention,
         'memory_gate': memory_gate, 'nb_classes': nb_classes, 'dropout': dropout})

    net = MACNetwork(params)

    import torch
    import numpy as np
    from problems.image_text_to_class.clevr import DataTuple, ImageTextTuple

    batch_size = 64
    embedded_dim = 300
    images = torch.from_numpy(np.random.binomial(n=1, p=0.5, size=(batch_size, 1024, 14, 14))).type(app_state.dtype)
    questions = torch.from_numpy(np.random.binomial(n=1, p=0.5, size=(batch_size, 15, embedded_dim))).type(
        app_state.dtype)
    answers = torch.from_numpy(np.random.randint(low=0, high=nb_classes, size=(batch_size, 1))).type(app_state.dtype)
    questions_len = [15] * batch_size

    # construct data_tuple
    image_text_tuple = ImageTextTuple(images, questions)
    inner_data_tuple = (image_text_tuple, questions_len)
    data_tuple = DataTuple(inner_data_tuple, answers)

    # Test base model.
    logits = net(data_tuple)