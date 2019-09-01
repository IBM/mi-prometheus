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
model.py:

    - Implementation of the ``VWM`` model, reusing the different units implemented in separated files.
"""
__author__ = "Vincent Albouy, T.S. Jayram, Tomasz Kornuta"

import nltk

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pylab
import matplotlib.animation
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import matplotlib.lines as lines

from miprometheus.models.model import Model
from miprometheus.models.vwm_model.utils_VWM import linear

from miprometheus.models.vwm_model.question_encoder import QuestionEncoder
from miprometheus.models.vwm_model.question_driven_controller import QuestionDrivenController
from miprometheus.models.vwm_model.image_encoder import ImageEncoder
from miprometheus.models.vwm_model.vwm_cell import VWMCell
from miprometheus.models.vwm_model.output_unit import OutputUnit

# needed for nltk.word.tokenize - do it once!
nltk.download('punkt')


class VWM(Model):
    """
    Implementation of the entire ``VWM`` network.
    """

    def __init__(self, params, problem_default_values_=None):
        """
        Constructor for the ``VWM`` network.

        :param params: dict of parameters (read from configuration ``.yaml`` file).
        :type params: utils.ParamInterface

        :param problem_default_values_: default values coming from the ``Problem`` class.
        :type problem_default_values_: dict
        """

        if problem_default_values_ is None:
            problem_default_values_ = {}

        # call base constructor
        super(VWM, self).__init__(params, problem_default_values_)

        # parse params dict
        self.dim = params['dim']
        self.embed_hidden = params['embed_hidden']  # embedding dimension
        self.max_step = params['max_step']
        self.dropout_param = params['dropout']
        self.slot = params['slot']
        self.nb_classes_pointing = params['classes']
        self.words_embed_length = params['words_embed_length']
        self.nwords = params['nwords']

        # Maximum number of embeddable words.
        self.vocabulary_size = problem_default_values_['embed_vocab_size']

        # Get dtype.
        self.dtype = self.app_state.dtype

        try:
            self.nb_classes = problem_default_values_['nb_classes']
            self.nb_classes_pointing = problem_default_values_['nb_classes_pointing']
        except KeyError:
            self.logger.warning("Couldn't retrieve one or more value(s) from problem_default_values_.")

        self.name = 'VWM'

        # instantiate units
        self.question_encoder = QuestionEncoder(
            self.vocabulary_size, dim=self.dim, embedded_dim=self.embed_hidden)

        self.question_driven_controller = QuestionDrivenController(
            self.dim, self.max_step, self.app_state.dtype)

        # instantiate units
        self.image_encoder = ImageEncoder(dim=self.dim)

        # linear layer for the projection of image features
        self.feature_maps_proj_layer = linear(self.dim, self.dim, bias=True)

        # initialize VWM Cell
        self.VWM_cell = VWMCell(dim=self.dim)

        # Create two separate output units.
        self.output_unit_answer = OutputUnit(dim=self.dim, nb_classes=self.nb_classes)

        # initialize hidden states for mac cell control states and memory states
        self.mem_0 = torch.nn.Parameter(torch.zeros(1, self.dim).type(self.app_state.dtype))
        self.control_0 = torch.nn.Parameter(torch.zeros(1, self.dim).type(self.app_state.dtype))

        self.dropout_layer = torch.nn.Dropout(self.dropout_param)

        self.frame_history = []

    def forward(self, data_dict):
        """
        Forward pass of the ``VWM`` network. Calls first the
        ``ImageEncoder, QuestionEncoder``, then the recurrent
        VWM cells and finally the ```OutputUnit``.

        :param data_dict: input data batch.
        :type data_dict: utils.DataDict

        :return: Predictions of the model.
        """

        # Change the order of image dimensions, so we will loop over
        # dimension 0: sequence elements.
        images = data_dict['images']
        images = images.permute(1, 0, 2, 3, 4)

        # Get batch size and length of image sequence.
        seq_len = images.size(0)
        batch_size = images.size(1)

        # Get and procecss questions.
        questions = data_dict['questions']

        # Get questions size of all batch elements.
        questions_length = questions.size(1)

        # Convert questions length into a tensor
        questions_length = torch.from_numpy(np.array(questions_length))

        # Create placeholders for logits.
        logits_answer = torch.zeros(
            (batch_size, seq_len, self.nb_classes),
            requires_grad=False).type(self.dtype)

        logits_pointing = torch.zeros(
            (batch_size, seq_len, self.nb_classes_pointing),
            requires_grad=False).type(self.dtype)

        # Apply dropout to VWM cell control_state_init states and summary object states
        control_state_init = self.control_0.expand(batch_size, -1)
        control_state_init = self.dropout_layer(control_state_init)

        summary_object_init = self.mem_0.expand(batch_size, -1)
        summary_object_init = self.dropout_layer(summary_object_init)

        # initialize empty memory
        visual_working_memory = (0.01 * torch.ones(
            batch_size, self.slot, self.dim)).type(self.app_state.dtype)

        # initialize read head at first slot position
        write_head = torch.zeros(batch_size, self.slot).type(self.app_state.dtype)
        write_head[:, 0] = 1

        self.frame_history = []

        # question encoder
        contextual_words, question_encoding = self.question_encoder(
            questions, questions_length)

        control_state = control_state_init
        control_history = []

        eps_tensor = torch.tensor([torch.finfo().eps]).type(self.app_state.dtype)
        for step in range(self.max_step):
            control_state, *control_other = self.question_driven_controller(
                step, contextual_words, question_encoding, control_state, eps_tensor)

            control_history.append((control_state, *control_other, step))

        # Loop over all elements along the SEQUENCE dimension.
        for f in range(images.size(0)):

            # RESET OF CONTROL and SUMMARY OBJECT
            summary_object = summary_object_init

            # image encoder
            feature_maps = self.image_encoder(images[f])
            feature_maps_proj = self.feature_maps_proj_layer(feature_maps)

            # state history fo vizualisation
            self.VWM_cell.cell_history = []

            # recurrent VWM cells
            for step in range(self.max_step):
                summary_object, visual_working_memory, write_head = self.VWM_cell(
                    control_history[step], feature_maps, feature_maps_proj,
                    summary_object, visual_working_memory, write_head)

            # print('Done with frame')
            # input('pp')

            # save state history
            self.frame_history.append(self.VWM_cell.cell_history)

            # output unit
            logits_answer[:, f, :] = self.output_unit_answer(question_encoding, summary_object)

        return logits_answer, logits_pointing

    @staticmethod
    def generate_figure_layout():
        """
        Generate a figure layout for the attention visualization (done in \
        ``MACNetwork.plot()``)

        :return: figure layout.

        """

        params = {'axes.titlesize': 'xx-large',
                  'axes.labelsize': 'xx-large',
                  'xtick.labelsize': 'x-large',
                  'ytick.labelsize': 'large',
                  }

        matplotlib.pylab.rcParams.update(params)

        # Prepare "generic figure template".
        # Create figure object.
        fig = Figure()

        ######################################################################
        # Top: Header section.
        # Create a specific grid.
        gs_header = GridSpec(1, 20)
        gs_header.update(wspace=0.00, hspace=0.00, bottom=0.9, top=0.901, left=0.01, right=0.99)
        _ = fig.add_subplot(gs_header[0, 0:2])
        _ = fig.add_subplot(gs_header[0, 2:14])
        _ = fig.add_subplot(gs_header[0, 14:16])
        _ = fig.add_subplot(gs_header[0, 16:20])

        ######################################################################
        # Top-center: Question + time context section.
        # Create a specific grid.
        gs_top = GridSpec(1, 24)
        gs_top.update(wspace=0.05, hspace=0.00, bottom=0.8, top=0.83, left=0.01, right=0.99)

        # Question with attention.
        ax_attention_question = fig.add_subplot(gs_top[0, 0:19], frameon=False)
        ax_attention_question.xaxis.set_major_locator(ticker.MaxNLocator(nbins=25))
        ax_attention_question.yaxis.set_major_locator(ticker.NullLocator())
        ax_attention_question.set_title('Question')

        # Time gate ;)
        ax_context = fig.add_subplot(gs_top[0, 20:24])
        ax_context.yaxis.set_major_locator(ticker.NullLocator())
        ax_context.xaxis.set_major_locator(ticker.FixedLocator([0, 1, 2, 3]))
        ax_context.set_xticklabels(['Last', 'Latest', 'Now', 'None'], horizontalalignment='left',
                                   rotation=-45, rotation_mode='anchor')
        ax_context.set_title('Time Context')

        ######################################################################
        # Bottom left: Image section.
        # Create a specific grid.
        gs_bottom_left = GridSpec(1, 2)
        gs_bottom_left.update(wspace=0.04, hspace=0.0, bottom=0.02, top=0.63, left=0.01, right=0.44)

        # Image.
        ax_image = fig.add_subplot(gs_bottom_left[0, 0])
        ax_image.xaxis.set_major_locator(ticker.NullLocator())
        ax_image.yaxis.set_major_locator(ticker.NullLocator())
        ax_image.set_title('Image')

        # Attention over the image.
        ax_attention_image = fig.add_subplot(gs_bottom_left[0, 1])
        ax_attention_image.xaxis.set_major_locator(ticker.NullLocator())
        ax_attention_image.yaxis.set_major_locator(ticker.NullLocator())
        ax_attention_image.set_title('Visual Attention')

        ######################################################################
        # Bottom Center: gates section.
        # Create a specific grid - for gates.
        gs_bottom_center = GridSpec(2, 1)
        gs_bottom_center.update(wspace=0.0, hspace=1, bottom=0.27, top=0.45, left=0.48, right=0.52)

        # Image gate.
        ax_image_match = fig.add_subplot(gs_bottom_center[0, 0])
        ax_image_match.xaxis.set_major_locator(ticker.NullLocator())
        ax_image_match.yaxis.set_major_locator(ticker.NullLocator())
        ax_image_match.set_title('Image Match')

        # Image gate.
        ax_memory_match = fig.add_subplot(gs_bottom_center[1, 0])
        ax_memory_match.xaxis.set_major_locator(ticker.NullLocator())
        ax_memory_match.yaxis.set_major_locator(ticker.NullLocator())
        ax_memory_match.set_title('Memory Match')

        ######################################################################
        # Bottom Right: Memory section.
        # Create a specific grid.
        gs_bottom_right = GridSpec(1, 20)
        gs_bottom_right.update(wspace=0.5, hspace=0.0, bottom=0.02, top=0.63, left=0.52, right=0.99)

        # Read attention.
        ax_attention_history = fig.add_subplot(gs_bottom_right[0, 3])
        ax_attention_history.xaxis.set_major_locator(ticker.NullLocator())
        ax_attention_history.set_ylabel('Memory Addresses')
        ax_attention_history.set_title('Read Head')

        # Memory
        ax_visual_working_memory = fig.add_subplot(gs_bottom_right[0, 4:18])
        ax_visual_working_memory.xaxis.set_major_locator(ticker.NullLocator())
        ax_visual_working_memory.yaxis.set_ticklabels([])
        ax_visual_working_memory.set_title('Working Memory')

        # Write attention.
        ax_wt = fig.add_subplot(gs_bottom_right[0, 18])
        ax_wt.xaxis.set_major_locator(ticker.NullLocator())
        ax_wt.yaxis.set_ticklabels([])
        ax_wt.set_title('Write Head')

        # Lines between sections.
        l1 = lines.Line2D([0, 1], [0.88, 0.88], transform=fig.transFigure,
                          figure=fig, color='black')
        l2 = lines.Line2D([0, 1], [0.68, 0.68], transform=fig.transFigure,
                          figure=fig, color='black')
        l3 = lines.Line2D([0.5, 0.5], [0.0, 0.25], transform=fig.transFigure,
                          figure=fig, color='black')
        l4 = lines.Line2D([0.5, 0.5], [0.49, 0.68], transform=fig.transFigure,
                          figure=fig, color='black')
        fig.lines.extend([l1, l2, l3, l4])

        # Set layout.
        # fig.set_tight_layout(True)

        return fig

    def plot(self, data_dict, logits, sample=0):
        """
        Visualize the attention weights (``ControlUnit`` & ``ReadUnit``) on the \
        question & feature maps. Dynamic visualization throughout the reasoning \
        steps is possible.
        :param data_dict: DataDict({'images','questions', 'questions_length',
        'questions_string', 'questions_type', \
        'targets', 'targets_string', 'index', 'prediction_string'})
        :type data_dict: utils.DataDict
        :param logits: Prediction of the model.
        :type logits: torch.tensor
        :param sample: Index of sample in batch (Default: 0)
        :type sample: int
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
        s_answers = data_dict['answers_string']
        vocab = data_dict['vocab']
        images = data_dict['images']
        tasks = data_dict['tasks']
        mask_pointing = data_dict['masks_pnt']

        # tokenize question string using same processing as in the problem
        question_words = s_questions[sample][0]
        words = nltk.word_tokenize(question_words)

        color = 'inferno'

        # get images dimensions
        width = images.size(3)
        height = images.size(4)

        # get task name
        tasks = tasks[sample]

        # ##################### CLASSIFICATION VISUALIZATION ####################

        if mask_pointing.sum() == 0:

            # get prediction
            _, indices = torch.max(logits[0], 2)
            prediction = indices[sample]

            # Create figure template.
            fig = self.generate_figure_layout()

            # Get axes that artists will draw on.
            (ax_header_left_labels, ax_header_left, ax_header_right_labels, ax_header_right,
             ax_attention_question, ax_context,
             ax_image, ax_attention_image,
             ax_image_match, ax_memory_match,
             ax_attention_history, ax_visual_working_memory, ax_wt) = fig.axes

            # initiate list of artists frames
            frames = []

            # loop over the sequence of frames
            for i in range(images.size(1)):

                # get image sample
                image = images[sample][i] / 255

                # needs [W x H x Channels] for Matplolib.imshow
                image = image.permute(1, 2, 0)

                # get answer and prediction strings
                pred = vocab[prediction[i]]
                ans = s_answers[sample][i]

                # loop over the k reasoning steps
                for (step, visual_attention, control_attention, visual_working_memory,
                     read_head, image_match, memory_match, write_head,
                     temporal_class_weights) in self.frame_history[i]:
                    # preprocess attention image, reshape
                    attention_size = int(np.sqrt(visual_attention.size(-1)))

                    # attention mask has size [batch_size x 1 x(H*W)]
                    attention_mask = visual_attention.view(-1, 1, attention_size, attention_size)

                    # upsample attention mask
                    m = torch.nn.Upsample(
                        size=[width, height], mode='bilinear', align_corners=True)
                    up_sample_attention_mask = m(attention_mask)
                    attention_mask = up_sample_attention_mask[sample, 0]

                    # norm = matplotlib.pylab.Normalize(0, 1)
                    # norm2 = matplotlib.pylab.Normalize(0, 4)

                    # Create "Artists" drawing data on "ImageAxes".
                    artists = []
                    # Tell artists what to do:

                    ######################################################################
                    # Set header.
                    ax_header_left_labels.axis('off')
                    artists.append(ax_header_left_labels.text(
                        0, 1.0,
                        'Question:         ' +
                        '\nPrediction: ' +
                        '\nGround Truth:     ',
                        fontsize='x-large'
                    ))
                    ax_header_left.axis('off')
                    artists.append(ax_header_left.text(
                        0, 1.0,
                        question_words +
                        '\n' + pred +
                        '\n' + ans,
                        fontsize='x-large',
                        weight='bold'))

                    ax_header_right_labels.axis('off')
                    artists.append(ax_header_right_labels.text(
                        0, 1.0,
                        'Frame: ' +
                        '\nReasoning Step:   ' +
                        '\nQuestion Type:    ',
                        fontsize='x-large'
                    ))

                    ax_header_right.axis('off')
                    artists.append(ax_header_right.text(
                        0, 1.0,
                        str(i) +
                        '\n' + str(step) +
                        '\n' + tasks,
                        fontsize='x-large',
                        weight='bold'))

                    ######################################################################
                    # Top-center: Question + time context section.
                    # Set words for question attention.
                    # ax_attention_question.xaxis.set_major_locator(
                    # ticker.MaxNLocator(nbins=len(words)))
                    # NOT WORKING AS number of ticks != number of cells! :]
                    ax_attention_question.set_xticklabels(words, horizontalalignment='left', rotation=-45, rotation_mode='anchor')

                    params = {'edgecolor': 'black', 'linewidth': 1.4e-3}

                    artists.append(ax_attention_question.pcolormesh(
                        control_attention[sample][..., None, :], vmin=0.0, vmax=1.0, **params))

                    # Time context.
                    # temporal_class_weights given by order now, last, latest, none
                    # visualization in different order last, latest, now, none
                    artists.append(ax_context.pcolormesh(
                        temporal_class_weights[[[sample]], [[1, 2, 0, 3]]], vmin=0.0, vmax=1.0, **params))

                    ######################################################################
                    # Bottom left: Image section.
                    artists.append(ax_image.imshow(image, interpolation='nearest', aspect='auto'))

                    # Two artists painting on the same figure - image + attention.
                    artists.append(ax_attention_image.imshow(
                        image, interpolation='nearest', aspect='auto'))
                    artists.append(ax_attention_image.imshow(
                        attention_mask,
                        interpolation='nearest',
                        aspect='auto',
                        alpha=0.5,
                        cmap=color))

                    ######################################################################
                    # Bottom center: gates section.

                    # Image gate.
                    artists.append(ax_image_match.pcolormesh(
                        image_match[[sample], None], vmin=0.0, vmax=1.0, **params))

                    # Memory gate.
                    artists.append(ax_memory_match.pcolormesh(
                        memory_match[[sample], None], vmin=0.0, vmax=1.0, **params))

                    ######################################################################
                    # Bottom Right: Memory section.

                    params2 = {'edgecolor': 'black', 'linewidth': 1.4e-4}
                    artists.append(ax_visual_working_memory.pcolormesh(
                        visual_working_memory[sample], **params2))

                    artists.append(ax_attention_history.pcolormesh(
                        read_head[sample][..., None], vmin=0.0, vmax=1.0, **params))

                    artists.append(ax_wt.pcolormesh(
                        write_head[sample][..., None], vmin=0.0, vmax=1.0, **params))

                    # Add "frames" to artist list
                    frames.append(artists)

            # Plot figure and list of frames.
            self.plotWindow.update(fig, frames)

        else:
            print("Visualization for pointing NOT OPERATIONAL!")
            exit(-10)

            # ################## POINTING VISUALIZATION #######################

            # get distribution
            softmax_pointing = nn.Softmax(dim=1)
            preds_pointing = softmax_pointing(logits[1])

            # Create figure template.
            fig = self.generate_figure_layout()

            # Get axes that artists will draw on.
            (ax_image, ax_attention_image, ax_attention_question, ax_step) = fig.axes

            # initiate list of artists frames
            frames = []

            # loop over the seqence of frames
            for i in range(images.size(1)):

                # get image sample
                image = images[sample][i]

                # needs [W x H x Channels] for Matplolib.imshow
                image = image.permute(1, 2, 0) / 255

                # loop over the k reasoning steps
                for step, (attention_mask, control_attention) in zip(
                        range(self.max_step), self.frame_history[i]):
                    # upsample attention mask
                    original_grid_size = 7
                    preds_pointing = preds_pointing.view(
                        images.size(0), images.size(1), original_grid_size, -1)
                    mm = torch.nn.Upsample(size=[width, height], mode='bilinear')
                    up_sample_preds_pointing = mm(preds_pointing)
                    up_sample_preds_pointing = up_sample_preds_pointing[sample][i]

                    # preprocess question, pick one sample number
                    control_attention = control_attention[sample]

                    # Create "Artists" drawing data on "ImageAxes".
                    num_artists = len(fig.axes) + 1
                    artists = [None] * num_artists

                    # set title labels
                    ax_image.set_title(
                        'COG image ' + str(i))
                    ax_attention_question.set_xticklabels(
                        ['h'] + words, rotation='vertical', fontsize=10)
                    ax_step.axis('off')
                    ax_attention_image.set_title(
                        'Pointing Distribution:')

                    # Tell artists what to do:
                    artists[0] = ax_image.imshow(
                        image, interpolation='nearest', aspect='auto')
                    artists[1] = ax_attention_image.imshow(
                        image, interpolation='nearest', aspect='auto')
                    artists[2] = ax_attention_image.imshow(
                        up_sample_preds_pointing.detach().numpy(),
                        interpolation='nearest',
                        aspect='auto',
                        alpha=0.5,
                        cmap='Blues')
                    artists[3] = ax_attention_question.imshow(
                        control_attention,
                        interpolation='nearest', aspect='auto', cmap='Reds')
                    artists[4] = ax_step.text(
                        0, 0.5, 'Reasoning step index: ' + str(
                            step) + ' | Question type: ' + tasks,
                        fontsize=15)

                    # Add "frames" to artist list
                    frames.append(artists)

            # Plot figure and list of frames.
            self.plotWindow.update(fig, frames)

        # return self.plotWindow.is_closed
