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

    - Implementation of the ``VWMC`` network, reusing the different units implemented in separated files.
"""
__author__ = "Vincent Marois, Vincent Albouy, T.S. Jayram, Tomasz Kornuta"

import nltk
# needed for nltk.word.tokenize - do it once!
nltk.download('punkt')

import torch
import numpy as np
import matplotlib.pylab
import matplotlib.animation
from miprometheus.models.model import Model
import numpy as numpy
import torch.nn as nn
from miprometheus.utils.app_state import AppState
app_state = AppState()
from miprometheus.models.VWM_model.question_encoder import QuestionEncoder
from miprometheus.models.VWM_model.image_encoder import ImageEncoder
from miprometheus.models.VWM_model.VWM_cell import VWMCell
from miprometheus.models.VWM_model.output_unit import OutputUnit
from matplotlib.colors import LinearSegmentedColormap


class MACNetworkSequential(Model):
    """
    Implementation of the entire ``VWM`` network.
    """

    def __init__(self, params, problem_default_values_={}):
        """
        Constructor for the ``VWM`` network.

        :param params: dict of parameters (read from configuration ``.yaml`` file).
        :type params: utils.ParamInterface

        :param problem_default_values_: default values coming from the ``Problem`` class.
        :type problem_default_values_: dict
        """

        # call base constructor
        super(MACNetworkSequential, self).__init__(params, problem_default_values_)

        # parse params dict
        self.dim = params['dim']
        self.embed_hidden = params['embed_hidden']  # embedding dimension
        self.max_step = params['max_step']
        self.self_attention = params['self_attention']
        self.memory_gate = params['memory_gate']
        self.dropout = params['dropout']
        self.memory_pass = params['memory_pass']
        self.control_pass = params['control_pass']
        self.slot =  params['slot']
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
        self.question_encoder = QuestionEncoder( self.vocabulary_size,self.dtype, self.words_embed_length, self.nwords, dim=self.dim, embedded_dim=self.embed_hidden )

        # instantiate units
        self.image_encoder = ImageEncoder(
            dim=self.dim)

        #initialize VWM Cell
        self.VWM_cell = VWMCell(
            dim=self.dim,
            max_step=self.max_step,
            dropout=self.dropout,
            slots=self.slot)

        # Create two separate output units.
        self.output_unit_answer = OutputUnit(dim=self.dim, nb_classes=self.nb_classes)


        # initialize hidden states for mac cell control states and memory states
        self.mem_0 = torch.nn.Parameter(torch.zeros(1, self.dim).type(app_state.dtype))
        self.control_0 = torch.nn.Parameter(
            torch.zeros(1, self.dim).type(app_state.dtype))


    def forward(self, data_dict, dropout=0.15):
        """
        Forward pass of the ``VWM`` network. Calls first the ``ImageEncoder, QuestionEncoder``, then the recurrent \
        VWM cells and finally the ```OutputUnit``.

        :param data_dict: input data batch.
        :type data_dict: utils.DataDict

        :param dropout: dropout rate.
        :type dropout: float

        :return: Predictions of the model.
        """

        # reset cell state history for visualization
        if self.app_state.visualize:
            self.VWM_cell.cell_state_history = []

        # Change the order of image dimensions, so we will loop over dimension 0: sequence elements.
        images = data_dict['images']
        images= images.permute(1, 0, 2, 3, 4)

        # Get batch size and length of image sequence.
        seq_len = images.size(0)
        batch_size = images.size(1)

        # Get and procecss questions.
        questions = data_dict['questions']

        # Get questions size of all batch elements.
        questions_length = questions.size(1)

        # Convert questions length into a tensor
        questions_length = torch.from_numpy(numpy.array(questions_length))

        # Create placeholders for logits.
        logits_answer = torch.zeros( (batch_size, seq_len, self.nb_classes), requires_grad=False).type(self.dtype)
        logits_pointing = torch.zeros( (batch_size, seq_len,self.nb_classes_pointing), requires_grad=False).type(self.dtype)

        # expand the hidden states to whole batch for mac cell control states and memory states
        control = self.control_0.expand(batch_size, self.dim)
        summary_object = self.mem_0.expand(batch_size, self.dim)
        control_mask = self.get_dropout_mask(control, self.dropout)
        memory_mask = self.get_dropout_mask(summary_object, self.dropout)
        control = control * control_mask
        summary_object= summary_object * memory_mask

        # initialize empty memory
        visual_working_memory \
            = torch.zeros(batch_size, self.slot ,self.dim).type(app_state.dtype)

        # initialize Wt_sequential at first slot position
        wt_sequential = torch.zeros(batch_size, self.slot).type(app_state.dtype)
        wt_sequential[:, 0] = 1

        self.cell_states=[]

        # question encoder
        contextual_word_encoding, question_encoding = self.question_encoder(questions, questions_length)

        # Loop over all elements along the SEQUENCE dimension.
        for f in range(images.size(0)):

            #RESET OF CONTROL and SUMMARY OBJECT
            new_summary_object = summary_object
            new_control_state = control

            # image encoder
            feature_maps= self.image_encoder(images[f])

            #state history fo vizualisation
            state_history=[]

            # recurrent VWM cells
            for i in range(self.max_step):
                new_summary_object, new_control_state, state_history, last_visual_attention, \
                visual_working_memory, wt_sequential \
                    = self.VWM_cell(contextual_word_encoding, question_encoding,
                                    feature_maps, new_control_state, new_summary_object,
                                    visual_working_memory, wt_sequential, state_history, step=i)




            # save state history
            self.cell_states.append(state_history)


            # output unit
            logits_answer[:, f, :] = self.output_unit_answer(last_visual_attention, question_encoding, new_summary_object)


        return logits_answer, logits_pointing

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

        ######################################################################
        # Top: Statistics section.
        #ax_step = fig.add_subplot(gs[0, :])
        #ax_step.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        #ax_step.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        #ax_step.axis('off')

        # We will use that for displaying "statistics" section.
        #statistics_text = 'Frame: ' + ' Reasoning step: '  + \
        #    '\nQuestion type: ' + \
        #    '\nPredicted Answer: ' + ' Ground Truth: ' + '\n'
        #fig.suptitle(statistics_text, fontsize=14)

        ######################################################################
        # Top-center: Question + time context section.
        # Create a specific grid.
        gs_top = matplotlib.gridspec.GridSpec(1, 6)
        gs_top.update(wspace=0.05, hspace=0.00, bottom=0.7, top=0.75, left=0.05, right=0.95)
        
        # Question with attention.
        ax_attention_question = fig.add_subplot(gs_top[0, 0:5])
        ax_attention_question.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=25))
        ax_attention_question.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        #ax_attention_question.set_xticklabels(25*[''], rotation=-45, fontsize=10)
        ax_attention_question.set_title('Question')

        # Time gate ;)
        ax_context = fig.add_subplot(gs_top[0, 5])
        ax_context.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax_context.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0,1,2,3]))
        ax_context.set_xticklabels(['Now','Last','Latest','None'], rotation=-45, fontsize=12)
        ax_context.set_title('Time Context')

        ######################################################################
        # Bottom left: Image section.
        # Create a specific grid.
        gs_bottom_left = matplotlib.gridspec.GridSpec(1, 2)
        gs_bottom_left.update(wspace=0.01, hspace=0.0, bottom=0.1, top=0.6, left=0.05, right=0.46)

        # Image.
        ax_image = fig.add_subplot(gs_bottom_left[0, 0])
        ax_image.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_image.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_image.set_ylabel('Height [px]', fontsize=8)
        ax_image.set_xlabel('Width [px]', fontsize=8)
        ax_image.set_title('Image')

        # Attention over the image.
        ax_attention_image = fig.add_subplot(gs_bottom_left[0, 1])
        ax_attention_image.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_attention_image.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_attention_image.set_xlabel('Width [px]', fontsize=8)
        ax_attention_image.set_title('Visual Attention')

        ######################################################################
        # Bottom Center: gates section.
        # Create a specific grid - for gates.
        gs_bottom_center = matplotlib.gridspec.GridSpec(2, 1)
        gs_bottom_center.update(wspace=0.0, hspace=1, bottom=0.42, top=0.60, left=0.48, right=0.52)

        # Image gate.
        ax_image_gate = fig.add_subplot(gs_bottom_center[0, 0])
        ax_image_gate.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax_image_gate.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax_image_gate.set_title('Image Gate')

        # Image gate.
        ax_memory_gate = fig.add_subplot(gs_bottom_center[1, 0])
        ax_memory_gate.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax_memory_gate.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax_memory_gate.set_title('Memory Gate')

        ######################################################################
        # Bottom Right: Memory section.
        # Create a specific grid.
        gs_bottom_right = matplotlib.gridspec.GridSpec(1, 10)
        gs_bottom_right.update(wspace=0.5, hspace=0.0, bottom=0.1, top=0.6, left=0.54, right=0.95)

        # Read attention.
        ax_attention_history = fig.add_subplot(gs_bottom_right[0, 0])
        ax_attention_history.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax_attention_history.set_ylabel('Memory Addresses', fontsize=8)
        ax_attention_history.set_title('Read Attention')

        # Memory
        ax_history = fig.add_subplot(gs_bottom_right[0, 1:9])
        ax_history.set_xlabel('Memory Content', fontsize=8)
        ax_history.set_title('Working Memory')


        # Write attention.
        ax_wt = fig.add_subplot(gs_bottom_right[0, 9])
        ax_wt.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax_wt.set_title('Write Attention')

        # Set layout.
        #fig.set_tight_layout(True)

        return fig

    def grayscale_cmap(self,cmap):
        """Return a grayscale version of the given colormap"""
        cmap = matplotlib.pylab.cm.get_cmap(cmap)
        colors = cmap(np.arange(cmap.N))

        # convert RGBA to perceived grayscale luminance
        # cf. http://alienryderflex.com/hsp.html
        RGB_weight = [0.299, 0.587, 0.114]
        luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
        colors[:, :3] = luminance[:, np.newaxis]

        return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)

    def view_colormap(self, cmap):
        """Plot a colormap with its grayscale equivalent"""
        cmap = matplotlib.pylab.cm.get_cmap(cmap)
        colors = cmap(np.arange(cmap.N))

        cmap = self.grayscale_cmap(cmap)
        grayscale = cmap(np.arange(cmap.N))

        fig, ax = matplotlib.pylab.subplots(2, figsize=(6, 2),
                               subplot_kw=dict(xticks=[], yticks=[]))
        ax[0].imshow([colors], extent=[0, 10, 0, 1])
        ax[1].imshow([grayscale], extent=[0, 10, 0, 1])



    def plot(self, data_dict, logits, sample=0):
        """
        Visualize the attention weights (``ControlUnit`` & ``ReadUnit``) on the \
        question & feature maps. Dynamic visualization throughout the reasoning \
        steps is possible.
        :param data_dict: DataDict({'images','questions', 'questions_length', 'questions_string', 'questions_type', \
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
        words = s_questions[sample][0]
        words = nltk.word_tokenize(words)

        color='plasma'

        # get images dimensions
        width = images.size(3)
        height = images.size(4)

        # get task name
        tasks = tasks[sample]

        ###################### CLASSIFICATION VISUALIZATION ####################

        if mask_pointing.sum()==0:

            # get prediction
            _, indices = torch.max(logits[0], 2)
            prediction = indices[sample]

            # Create figure template.
            fig = self.generate_figure_layout()

            # Get axes that artists will draw on.
            (ax_attention_question, ax_context, ax_image, ax_attention_image, ax_image_gate, ax_memory_gate, ax_attention_history, ax_history,ax_wt) = fig.axes

            #initiate list of artists frames
            frames = []

            #loop over the sequence of frames
            for i in range(images.size(1)):

                #get image sample
                image = images[sample][i]/255

                # needs [W x H x Channels] for Matplolib.imshow
                image = image.permute(1, 2, 0)

                #get answer and prediction strings
                pred = vocab[prediction[i]]
                ans = s_answers[sample][i]

                #loop over the k reasoning steps
                for step, (attention_mask, attention_question, history, W, gmem, gkb, Wt_seq , context) in zip(
                        range(self.max_step), self.cell_states[i]):

                    print("gmem: ", gmem[sample])
                    print("gkb: ", gkb[sample])
                    
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

                    norm = matplotlib.pylab.Normalize(0, 1)
                    #norm2 = matplotlib.pylab.Normalize(0, 4)

                    # Create "Artists" drawing data on "ImageAxes".
                    artists = []

                    # Tell artists what to do:
                    
                    # Set title.
                    statistics_text = 'Frame: ' + str(i) + ' Reasoning step: ' + str(step) + \
                        '\nQuestion type: ' + tasks + \
                        '\nPredicted Answer: ' + pred + ' Ground Truth: ' + ans + '\n'
                    #fig.suptitle(statistics_text, fontsize=14)
                    at = matplotlib.offsetbox.AnchoredText(statistics_text,
                        loc='upper left', prop=dict(size=8), frameon=True,
                        )
                    #at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
                    artists.append(at)

                    ######################################################################
                    # Top-center: Question + time context section.
                    # Set words for question attention.
                    ax_attention_question.set_xticklabels(['h'] + words, rotation=-45, fontsize=12)
                    artists.append(ax_attention_question.imshow(
                        attention_question.unsqueeze(1).transpose(1, 0),
                        interpolation='nearest', aspect='auto', cmap=color, norm=norm))

                    # Time context.
                    artists.append(ax_context.imshow(
                        context[sample], interpolation='nearest', cmap=color, norm=norm, aspect='auto'))
    
                    ######################################################################
                    # Bottom left: Image section.
                    artists.append(ax_image.imshow(image, interpolation='nearest', aspect='auto'))

                    # Two artists painting on the same figure - image + attention.
                    artists.append(ax_attention_image.imshow(image, interpolation='nearest', aspect='auto'))
                    artists.append(ax_attention_image.imshow(
                        attention_mask,
                        interpolation='nearest',
                        aspect='auto',
                        alpha=0.5,
                        cmap=color))

                    ######################################################################
                    # Bottom center: gates section.
                    
                    # Image gate.
                    artists.append(ax_image_gate.imshow(
                        [[ gkb[sample] ]], interpolation='nearest', cmap=color, norm=norm, aspect='auto'))
                    
                    # Memory gate.
                    artists.append(ax_memory_gate.imshow(
                        [[ gmem[sample] ]], interpolation='nearest', cmap=color, norm=norm, aspect='auto'))

                    ######################################################################
                    # Bottom Right: Memory section.

                    artists.append(ax_history.imshow(
                        history[sample], interpolation='nearest', aspect='auto', cmap=color, norm=norm ))
                    #    history[sample], interpolation='nearest', aspect='auto', cmap=color, norm=norm2  )) WHY DIFFERENT NORMALIZATION??

                    artists.append(ax_attention_history.imshow(
                        W[sample].unsqueeze(1), interpolation='nearest',cmap=color, norm=norm , aspect='auto'))

                    artists.append(ax_wt.imshow(
                        Wt_seq[sample].transpose(1,0), interpolation='nearest', cmap=color, norm=norm, aspect='auto'))

                    # Add "frames" to artist list
                    frames.append(artists)

            # Plot figure and list of frames.
            self.plotWindow.update(fig, frames)

        else:

            # NOT OPERATIONAL!!!!

            ################### POINTING VISUALIZATION #######################

            #get distribution
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

                #needs [W x H x Channels] for Matplolib.imshow
                image = image.permute(1, 2, 0)/255

                # loop over the k reasoning steps
                for step, (attention_mask, attention_question) in zip(
                        range(self.max_step), self.cell_states[i]):



                    # upsample attention mask
                    original_grid_size=7
                    preds_pointing = preds_pointing.view(images.size(0),images.size(1),original_grid_size, -1)
                    mm =torch.nn.Upsample(size=[width, height] , mode= 'bilinear')
                    up_sample_preds_pointing = mm(preds_pointing)
                    up_sample_preds_pointing = up_sample_preds_pointing[sample][i]

                    # preprocess question, pick one sample number
                    attention_question = attention_question[sample]

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
                        attention_question,
                        interpolation='nearest', aspect='auto', cmap='Reds')
                    artists[4] = ax_step.text(
                        0, 0.5, 'Reasoning step index: ' + str(
                            step) + ' | Question type: ' + tasks,
                        fontsize=15)

                    # Add "frames" to artist list
                    frames.append(artists)

            # Plot figure and list of frames.
            self.plotWindow.update(fig, frames)

        #return self.plotWindow.is_closed


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
