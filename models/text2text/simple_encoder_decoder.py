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

"""simple_encoder_decoder.py: Implementation of an Encoder-Decoder network for text2text problems (e.g. translation)
    Inspiration taken from the corresponding Pytorch tutorial.
    See https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html """
__author__ = "Vincent Marois "

import torch
import random

from models.text2text.encoder import EncoderRNN
from models.text2text.base_decoder import DecoderRNN
from models.text2text.attn_decoder import AttnDecoderRNN
from models.sequential_model import SequentialModel

# global tokens
PAD_token = 0
SOS_token = 1
EOS_token = 2


class SimpleEncoderDecoder(SequentialModel):
    """
    Sequence to Sequence model based on EncoderRNN & DecoderRNN.
    """

    def __init__(self, params):
        """
        Initializes the Encoder-Decoder network.

        :param params: dict containing the main parameters set:
            - max_length: maximal length of the input / output sequence of words: i.e, max length of the sentences
            to translate -> upper limit of seq_length
            - input_voc_size: should correspond to the length of the vocabulary set of the input language
            - hidden size: size of the embedding & hidden states vectors.
            - output_voc_size: should correspond to the length of the vocabulary set of the output language

        """
        # call base constructor
        super(SimpleEncoderDecoder, self).__init__(params)

        self.max_length = params['max_length']

        # parse params to create encoder
        self.input_voc_size = params['input_voc_size']
        self.hidden_size = params['hidden_size']
        self.encoder_bidirectional = params['encoder_bidirectional']

        # create encoder
        self.encoder = EncoderRNN(
            input_voc_size=self.input_voc_size,
            hidden_size=self.hidden_size,
            bidirectional=self.encoder_bidirectional,
            n_layers=1)

        # parse param to create decoder
        self.output_voc_size = params['output_voc_size']

        # create base decoder
        #self.decoder = DecoderRNN(hidden_size=self.hidden_size, output_voc_size=self.output_voc_size)

        # create attention decoder
        self.decoder = AttnDecoderRNN(
            self.hidden_size,
            self.output_voc_size,
            dropout_p=0.1,
            max_length=self.max_length,
            encoder_bidirectional=self.encoder_bidirectional)

        print('EncoderDecoderRNN (with Bahdanau attention) created.\n')

    def plot(self, data_tuple, predictions, sample_number=0):
        """
        Plot function to visualize the attention weights on the input sequence
        as the model is generating the output sequence.

        :param data_tuple: data_tuple: Data tuple containing input [BATCH_SIZE x SEQUENCE_LENGTH] and target sequences
        [BATCH_SIZE x SEQUENCE_LENGTH]

        :param predictions: logits as dict {'inputs_text', 'logits_text'}

        :param sample_number:


        """
        # Check if we are supposed to visualize at all.
        if not self.app_state.visualize:
            return False

        # Initialize timePlot window - if required.
        if self.plotWindow is None:
            from utils.time_plot import TimePlot
            self.plotWindow = TimePlot()

        # select 1 random sample in the batch and retrieve corresponding
        # input_text, logit_text, attention_weight
        batch_size = data_tuple.targets.shape[0]
        sample = random.choice(range(batch_size))

        # pred should be a dict {'inputs_text', 'logits_text'} created by
        # Translation.plot_processing()
        input_text = predictions['inputs_text'][sample].split()
        print('input sentence: ', predictions['inputs_text'][sample])
        target_text = predictions['logits_text'][sample]
        print('predicted translation:', target_text)
        attn_weights = self.decoder_attentions[sample].cpu().detach().numpy()

        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attn_weights)

        # set up axes
        ax.set_xticklabels([''] + input_text, rotation=90)
        ax.set_yticklabels([''] + target_text)

        # show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # Plot figure and list of frames.
        self.plotWindow.update(fig, frames=[[cax]])

        # Return True if user closed the window.
        return self.plotWindow.is_closed

    # global forward pass
    def forward(self, data_tuple):
        """
        Runs the network.

        :param data_tuple: (input_tensor, target_tensor) tuple

        :return: decoder outputs: of shape [target_length x output_voc_size] containing the probability distributions
        over the vocabulary set for each word in the target sequence.


        """
        # unpack data_tuple
        (inputs, targets) = data_tuple

        # get batch_size (dim 0)
        batch_size = inputs.size(0)

        # reshape tensors: from [batch_size x max_seq_length] to
        # [max_seq_length x batch_size]
        input_tensor = inputs.transpose(0, 1)
        target_tensor = targets.transpose(0, 1)

        # init encoder hidden states
        encoder_hidden = self.encoder.init_hidden(batch_size)

        # create placeholder for the encoder outputs -> will be passed to
        # attention decoder
        if self.encoder.bidirectional:
            encoder_outputs = torch.zeros(
                self.max_length,
                batch_size,
                (self.hidden_size *
                 2)).type(
                self.app_state.dtype)
        else:
            encoder_outputs = torch.zeros(
                self.max_length,
                batch_size,
                (self.hidden_size *
                 1)).type(
                self.app_state.dtype)

        # create placeholder for the attention weights -> for visualization
        self.decoder_attentions = torch.zeros(
            batch_size, self.max_length, self.max_length).type(self.app_state.dtype)

        # encoder manual loop
        for ei in range(self.max_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei].unsqueeze(-1), encoder_hidden)
            encoder_outputs[ei] = encoder_output.squeeze()

        # reshape encoder_outputs to be batch_size first: [max_length,
        # batch_size, *] -> [batch_size, max_length, *]
        encoder_outputs = encoder_outputs.transpose(0, 1)

        # decoder input : [batch_size x 1] initialized to the value of Start Of
        # String token
        decoder_input = torch.ones(batch_size, 1).type(
            self.app_state.LongTensor) * SOS_token

        # pass along the hidden states: shape [[(encoder.n_layers *
        # encoder.n_directions) x batch_size x hidden_size]]
        decoder_hidden = encoder_hidden

        # create placeholder for the decoder outputs -> will be the logits
        decoder_outputs = torch.zeros(
            self.max_length,
            batch_size,
            self.output_voc_size).type(
            self.app_state.dtype)

        if self.training:  # Teacher forcing: Feed the target as the next input
            for di in range(self.max_length):
                # base decoder
                #decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                # attention decoder
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)

                decoder_outputs[di] = decoder_output.squeeze()

                # Teacher forcing
                decoder_input = target_tensor[di].unsqueeze(-1)

        else:
            # Without teacher forcing: use its own predictions as the next
            # input
            for di in range(self.max_length):
                # base decoder
                #decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                # attention decoder
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_outputs[di] = decoder_output.squeeze()

                # save attention weights
                self.decoder_attentions[:, di, :] = decoder_attention.squeeze()

                # get most probable word as input of decoder for next iteration
                topv, topi = decoder_output.topk(k=1, dim=-1)

                # detach from history as input
                decoder_input = topi.view(batch_size, 1).detach()

                # TODO: The line below would stop inference when the next predicted word is the EOS token. This if
                # statement works for batch_size = 1, but how to generalize it to any size?
                # if decoder_input.item() == EOS_token:
                #    break

        return decoder_outputs.transpose(0, 1)


if __name__ == '__main__':
    # import lines for problem class
    import sys
    import os
    # TK: ok, what is going on in here...?

    # add problem folder to path for import
    sys.path.insert(0, os.path.normpath(os.path.join(
        os.getcwd(), '../../problems/seq_to_seq/text2text')))
    import translation as pb

    # instantiate problem
    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )

    params = {
        'batch_size': 64,
        'training_size': 0.90,
        'output_lang_name': 'fra',
        'max_sequence_length': 15,
        'eng_prefixes': eng_prefixes,
        'use_train_data': True,
        'data_folder': '~/data/language',
        'reverse': False}

    problem = pb.Translation(params)
    print('Problem successfully created.\n')

    # get size of vocabulary for input & output language
    input_voc_size = problem.input_lang.n_words
    output_voc_size = problem.output_lang.n_words

    # instantiate model with credible parameters
    model_params = {
        'max_length': 15,
        'input_voc_size': input_voc_size,
        'hidden_size': 256,
        'output_voc_size': output_voc_size,
        'encoder_bidirectional': True}
    net = SimpleEncoderDecoder(model_params)

    # generate a batch
    DataTuple, AuxTuple = problem.generate_batch()
    outputs = net(data_tuple=DataTuple)
    print('outputs: \n', outputs, '\n')

    # try to evaluate loss
    loss = problem.evaluate_loss(
        data_tuple=DataTuple, logits=outputs, aux_tuple=AuxTuple)
    print('loss: ', loss.item())
    loss.backward()

    bleu_score = problem.compute_BLEU_score(DataTuple, outputs, AuxTuple)
    print('BLEU score:', bleu_score)
