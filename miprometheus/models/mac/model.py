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
model.py:

    - Implementation of the ``MAC`` network, reusing the different units implemented in separated files.
    - Cf https://arxiv.org/abs/1803.03067 for the reference paper.


"""
__author__ = "Vincent Marois , Vincent Albouy"

import os
import nltk
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from miprometheus.models.model import Model
import numpy as numpy
import torch.nn as nn
from miprometheus.utils.app_state import AppState
app_state = AppState()
from miprometheus.models.mac.input_unit import InputUnit
from miprometheus.models.mac.mac_unit import MACUnit
from miprometheus.models.mac.output_unit import OutputUnit
from miprometheus.models.mac.image_encoding import ImageProcessing


class MACNetwork(Model):
    """
    Implementation of the entire ``MAC`` network.
    """

    def __init__(self, params, problem_default_values_={}):
        """
        Constructor for the ``MAC`` network.

        :param params: dict of parameters (read from configuration ``.yaml`` file).
        :type params: utils.ParamInterface

        :param problem_default_values_: default values coming from the ``Problem`` class.
        :type problem_default_values_: dict
        """

        # call base constructor
        super(MACNetwork, self).__init__(params, problem_default_values_)

        # parse params dict
        self.dim = params['dim']
        self.embed_hidden = params['embed_hidden']  # embedding dimension
        self.max_step = params['max_step']
        self.self_attention = params['self_attention']
        self.memory_gate = params['memory_gate']
        self.dropout = params['dropout']
        self.memory_pass = params['memory_pass']
        self.control_pass = params['control_pass']

        # Maximum number of embeddable words.
        self.vocabulary_size = problem_default_values_['embed_vocab_size']

        # Length of vectoral representation of each word.
        self.words_embed_length = 64

        # This should be the length of the longest sentence encounterable
        self.nwords = 24

        # Get dtype.
        self.dtype = self.app_state.dtype

        self.EmbedVocabulary(self.vocabulary_size,
                             self.words_embed_length)

        try:
            self.nb_classes = problem_default_values_['nb_classes']
            self.nb_classes_pointing = problem_default_values_['nb_classes_pointing']
        except KeyError:
            self.logger.warning("Couldn't retrieve one or more value(s) from problem_default_values_.")

        self.nb_classes_pointing=49

        self.name = 'MAC'

        # instantiate units
        self.input_unit = InputUnit(
            dim=self.dim, embedded_dim=self.embed_hidden)

        self.mac_unit = MACUnit(
            dim=self.dim,
            max_step=self.max_step,
            self_attention=self.self_attention,
            memory_gate=self.memory_gate,
            dropout=self.dropout)

        self.image_encoding = ImageProcessing(dim=512)

        self.output_unit = OutputUnit(dim=self.dim, nb_classes=self.nb_classes)
        self.output_unit_pointing = OutputUnit(dim=self.dim, nb_classes=self.nb_classes_pointing)


        self.data_definitions = {'images': {'size': [-1, 1024, 14, 14], 'type': [np.ndarray]},
                                 'questions': {'size': [-1, -1, -1], 'type': [torch.Tensor]},
                                 'questions_length': {'size': [-1], 'type': [list, int]},
                                 'targets': {'size': [-1, self.nb_classes], 'type': [torch.Tensor]}
                                 }

        ####### IMAGE PROCESSING ################

        # Number of channels in input Image
        self.image_channels = 3

        # CNN number of channels
        self.visual_processing_channels = [32, 64, 64, 128]

        # LSTM hidden units.
        self.lstm_hidden_units = 64

        # Input Image size
        self.image_size = [112, 112]

        # Visual memory shape. height x width.
        self.vstm_shape = np.array(self.image_size)
        for channel in self.visual_processing_channels:
            self.vstm_shape = np.floor((self.vstm_shape) / 2)
        self.vstm_shape = [int(dim) for dim in self.vstm_shape]

        # Number of GRU units in controller
        self.controller_output_size = 768

        self.VisualProcessing(self.image_channels,
                              self.visual_processing_channels,
                              self.lstm_hidden_units * 2,
                              self.controller_output_size * 2,
                              self.vstm_shape)

        # Initialize weights and biases
        # -----------------------------------------------------------------
        # Visual processing
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain('relu'))

        self.conv1.bias.data.fill_(0.01)
        self.conv2.bias.data.fill_(0.01)
        self.conv3.bias.data.fill_(0.01)
        self.conv4.bias.data.fill_(0.01)

        # initialize hidden states for mac cell control states and memory states
        self.mem_0 = torch.nn.Parameter(torch.zeros(1, self.dim).type(app_state.dtype))
        self.control_0 = torch.nn.Parameter(
            torch.zeros(1, self.dim).type(app_state.dtype))



    def forward(self, data_dict, dropout=0.15):
        """
        Forward pass of the ``MAC`` network. Calls first the ``InputUnit``, then the recurrent \
        MAC cells and finally the ```OutputUnit``.

        :param data_dict: input data batch.
        :type data_dict: utils.DataDict

        :param dropout: dropout rate.
        :type dropout: float

        :return: Predictions of the model.
        """

        # reset cell state history for visualization
        if self.app_state.visualize:
            self.mac_unit.cell_state_history = []

        # unpack data_dict
        images = data_dict['images']
        images= images.permute(1, 0, 2, 3, 4)



        #logits placeholders
        logits = torch.zeros(data_dict['images'].size(0), images.size(0), self.nb_classes)
        logits_pointing = torch.zeros(data_dict['images'].size(0), images.size(0),self.nb_classes_pointing )

        batch_size = data_dict['images'].size(0)

        # expand the hidden states to whole batch for mac cell control states and memory states
        control = self.control_0.expand(batch_size, self.dim)
        memory = self.mem_0.expand(batch_size, self.dim)
        control_mask = self.get_dropout_mask(control, self.dropout)
        memory_mask = self.get_dropout_mask(memory, self.dropout)
        control = control * control_mask
        memory = memory * memory_mask

        # expand the hidden states to whole batch for mac cell control states and memory states
        controls = [control]
        memories = [memory]

        for i in range(images.size(0)):

            #Cog like CNN - cf  cog  model

            x = self.conv1(images[i])
            x = self.maxpool1(x)
            x = nn.functional.relu(self.batchnorm1(x))
            x = self.conv2(x)
            x = self.maxpool2(x)
            x = nn.functional.relu(self.batchnorm2(x))
            x = self.conv3(x)
            x = self.maxpool3(x)
            x = nn.functional.relu(self.batchnorm3(x))
            x = self.conv4(x)
            # out_conv4 = self.conv4(out_batchnorm3)
            x = self.maxpool4(x)


            #get question from data dict

            questions = data_dict['questions']
            #print(questions.size())

            questions = self.forward_lookup2embed(questions)

            #get questions size of all batch elements
            questions_length = questions.size(1)

            #convert questions lenght into a tensor
            questions_length = torch.from_numpy(numpy.array(questions_length))

            # input unit
            img, kb_proj, lstm_out, h = self.input_unit(
            questions, questions_length, x)

            # recurrent MAC cells
            memory, controls, memories = self.mac_unit(lstm_out, h, img, kb_proj, controls, memories, self.control_pass, self.memory_pass )

            targets_reg = data_dict['targets_reg']
            targets_class = data_dict['targets_class']
            #print(targets_reg)
            #print(targets_class)

            # output unit
            logits[:,i,:] = self.output_unit(memory, h)
            logits_pointing[:,i,:] = self.output_unit_pointing(memory, h)


        return logits.cuda(), logits_pointing.cuda()

    @staticmethod
    def generate_figure_layout():
        """
        Generate a figure layout for the attention visualization (done in \
        ``MACNetwork.plot()``)

        :return: figure layout.

        """
        import matplotlib
        from matplotlib.figure import Figure

        params = {'axes.titlesize': 'large',
                  'axes.labelsize': 'large',
                  'xtick.labelsize': 'medium',
                  'ytick.labelsize': 'medium'}
        matplotlib.pylab.rcParams.update(params)

        # Prepare "generic figure template".
        # Create figure object.
        fig = Figure()

        # Create a specific grid for MAC.
        gs = matplotlib.gridspec.GridSpec(6, 2)

        # subplots: original image, attention on image & question, step index
        ax_image = fig.add_subplot(gs[2:6, 0])
        ax_attention_image = fig.add_subplot(gs[2:6, 1])
        ax_attention_question = fig.add_subplot(gs[0, :])
        ax_step = fig.add_subplot(gs[1, 0])

        # Set axis ticks
        ax_image.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_image.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_attention_image.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True))
        ax_attention_image.yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True))

        # question ticks
        ax_attention_question.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=40))

        ax_step.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_step.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        fig.set_tight_layout(True)

        return fig

    def forward_lookup2embed(self, questions):
        """
        Performs embedding of lookup-table questions with nn.Embedding.

        :param questions: Tensor of questions in lookup format (Ints)

        """

        out_embed = torch.zeros((questions.size(0), self.nwords, self.words_embed_length), requires_grad=False).type(
            self.dtype)
        for i, sentence in enumerate(questions):
            out_embed[i, :, :] = (self.Embedding(sentence))

        return out_embed

        # Embed vocabulary for all available task families
    def EmbedVocabulary(self, vocabulary_size, words_embed_length):
            """
            Defines nn.Embedding for embedding of questions into float tensors.

            :param vocabulary_size: Number of unique words possible.
            :type vocabulary_size: Int

            :param words_embed_length: Size of the vectors representing words post embedding.
            :type words_embed_length: Int

            """
            self.Embedding = nn.Embedding(vocabulary_size, words_embed_length, padding_idx=0)

    def VisualProcessing(self, in_channels, layer_channels, feature_control_len, spatial_control_len, output_shape):
        """
        Defines all layers pertaining to visual processing.

        :param in_channels: Number of channels in images in dataset. Usually 3 (RGB).
        :type in_channels: Int

        :param layer_channels: Number of feature maps in the CNN for each layer.
        :type layer_channels: List of Ints

        :param feature_control_len: Input size to the Feature Attention linear layer.
        :type feature_control_len: Int

        :param spatial_control_len: Input size to the Spatial Attention linear layer.
        :type spatial_control_len: Int

        :param output_shape: Output dimensions of feature maps of last layer.
        :type output_shape: Tuple of Ints

        """
        # Initial Norm
        # self.batchnorm0 = nn.BatchNorm2d(3)

        # First Layer
        self.conv1 = nn.Conv2d(in_channels, layer_channels[0], 3,
                               stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.maxpool1 = nn.MaxPool2d(2,
                                     stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.batchnorm1 = nn.BatchNorm2d(layer_channels[0])

        # Second Layer
        self.conv2 = nn.Conv2d(layer_channels[0], layer_channels[1], 3,
                               stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.maxpool2 = nn.MaxPool2d(2,
                                     stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.batchnorm2 = nn.BatchNorm2d(layer_channels[1])

        # Third Layer
        self.conv3 = nn.Conv2d(layer_channels[1], layer_channels[2], 3,
                               stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.maxpool3 = nn.MaxPool2d(2,
                                     stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.batchnorm3 = nn.BatchNorm2d(layer_channels[2])

        # Fourth Layer
        self.conv4 = nn.Conv2d(layer_channels[2], layer_channels[3], 3,
                               stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.maxpool4 = nn.MaxPool2d(2,
                                     stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.batchnorm4 = nn.BatchNorm2d(layer_channels[3])

        # Linear Layer
        self.cnn_linear1 = nn.Linear(layer_channels[3] * output_shape[0] * output_shape[1], 128)


    def plot(self, data_dict, logits, sample=0):
        """
        Visualize the attention weights (``ControlUnit`` & ``ReadUnit``) on the \
        question & feature maps. Dynamic visualization throughout the reasoning \
        steps is possible.

        :param data_dict: DataDict({'images','questions', 'questions_length', 'questions_string', 'questions_type', \
        'targets', 'targets_string', 'index','imgfiles', 'prediction_string'})
        :type data_dict: utils.DataDict

        :param logits: Prediction of the model.
        :type logits: torch.tensor

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
        # tokenize question string using same processing as in the problem
        # class
        words = nltk.word_tokenize(s_questions[sample])

        # Create figure template.
        fig = self.generate_figure_layout()
        # Get axes that artists will draw on.
        (ax_image, ax_attention_image, ax_attention_question, ax_step) = fig.axes

        # get the image
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
            m = torch.nn.Upsample(
                size=[width, height], mode='bilinear', align_corners=True)
            up_sample_attention_mask = m(attention_mask)
            attention_mask = up_sample_attention_mask[sample, 0]

            # preprocess question, pick one sample number
            attention_question = attention_question[sample]

            # Create "Artists" drawing data on "ImageAxes".
            num_artists = len(fig.axes) + 1
            artists = [None] * num_artists

            # set title labels
            ax_image.set_title(
                'CLEVR image: {}'.format(imgfiles[sample]))
            ax_attention_question.set_xticklabels(
                ['h'] + words, rotation='vertical', fontsize=10)
            ax_step.axis('off')

            # set axis attention labels
            ax_attention_image.set_title(
                'Predicted Answer: ' + prediction_string[sample] +
                ' [ proba: ' + str.format("{0:.3f}", proba_answer) + ']  ' +
                'Ground Truth: ' + answer_string[sample])

            # Tell artists what to do:
            artists[0] = ax_image.imshow(
                image, interpolation='nearest', aspect='auto')
            artists[1] = ax_attention_image.imshow(
                image, interpolation='nearest', aspect='auto')
            artists[2] = ax_attention_image.imshow(
                attention_mask,
                interpolation='nearest',
                aspect='auto',
                alpha=0.5,
                cmap='Reds')
            artists[3] = ax_attention_question.imshow(
                attention_question.transpose(1, 0),
                interpolation='nearest', aspect='auto', cmap='Reds')
            artists[4] = ax_step.text(
                0, 0.5, 'Reasoning step index: ' + str(step) + ' | Question type: ' + question_type[sample],
                fontsize=15)

            # Add "frame".
            frames.append(artists)

        # Plot figure and list of frames.
        self.plotWindow.update(fig, frames)

        return self.plotWindow.is_closed


    def get_dropout_mask(self, x, dropout):
        """
        Create a dropout mask to be applied on x.

        :param x: tensor of arbitrary shape to apply the mask on.
        :type x: torch.tensor

        :param dropout: dropout rate.
        :type dropout: float

        :return: mask.

        """
        # create a binary mask, where the probability of 1's is (1-dropout)
        mask = torch.empty_like(x).bernoulli_(
            1 - dropout).type(app_state.dtype)

        # normalize the mask so that the average value is 1 and not (1-dropout)
        mask /= (1 - dropout)

        return mask


if __name__ == '__main__':
    dim = 128
    embed_hidden = 64
    max_step = 12
    self_attention = True
    memory_gate = True
    nb_classes = 28
    dropout = 0.15

    from miprometheus.utils.app_state import AppState
    from miprometheus.utils.param_interface import ParamInterface
    from torch.utils.data import DataLoader
    from miprometheus.utils.param_interface import ParamInterface
    from miprometheus.problems.seq_to_seq.video_text_to_class.cog import COG
    import os
    import torch.optim as optim
    app_state = AppState()

    from miprometheus.problems import CLEVR

    params = ParamInterface()
    #problem_params.add_config_params({'settings': {'data_folder': '~/CLEVR_v1.0',
     #                                              'set': 'train', 'dataset_variant': 'CLEVR'},
#
     #                                 'images': {'raw_images': False,
     #                                            'feature_extractor': {'cnn_model': 'resnet101',
     #                                                                  'num_blocks': 4}},
#
    #                                  'questions': {'embedding_type': 'random', 'embedding_dim': 300}})

    # create problem
    #clevr_dataset = CLEVR(problem_params)
   # print('Problem {} instantiated.'.format(clevr_dataset.name))

    tasks = ['GoColor', ]
    params.add_config_params({'data_folder': os.path.expanduser('~/data/cog'),
                              'set': 'val',
                              'dataset_type': 'canonical',
                              'tasks': tasks})

    # Create problem - task Go
    cog_dataset = COG(params)

    # Get a sample - Go
    sample = cog_dataset[0]
    #print(repr(sample))
    print(sample['images'].size())
    print('hello')



    # instantiate DataLoader object
    batch_size=48
    #problem = DataLoader(clevr_dataset, batch_size=batch_size, collate_fn=clevr_dataset.collate_fn)
    problem = DataLoader(cog_dataset, batch_size=batch_size, collate_fn=cog_dataset.collate_fn)

    model_params = ParamInterface()
    model_params.add_config_params({'dim': dim,
                                    'memory_pass' : False,
                                    'control_pass' : False,
                                    'embed_hidden': embed_hidden,
                                    'max_step': 12,
                                    'self_attention': self_attention,
                                    'memory_gate': memory_gate,
                                    'dropout': dropout})

    model = MACNetwork(model_params, cog_dataset.default_values)
    print('Model {} instantiated.'.format(model.name))
    model.app_state.visualize = True

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # perform handshaking between MAC & CLEVR
    #model.handshake_definitions(clevr_dataset.data_definitions)

    # generate a batch
    for i_batch, sample in enumerate(problem):

        print('Sample # {} - {}'.format(i_batch, sample['images'].shape), type(sample))
        logits = model(sample)
        #print('logits  of size :', logits.size())
        loss=cog_dataset.evaluate_loss(sample,logits)
        acc=cog_dataset.calculate_accuracy(sample,logits)
        cog_dataset.get_acc_per_family(sample, logits)
        print(loss)
        print(acc)
        loss.backward()
        optimizer.step()
        #clevr_dataset.plot_preprocessing(sample, logits)
        #model.plot(sample, logits)
        #print(logits.shape)

    print('Unit test completed.')
