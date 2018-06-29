#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of an Encoder-Decoder network for text2text problems (e.g. translation)"""
__author__ = "Vincent Marois "

import torch
# Add path to main project directory
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..', '..'))

from misc.app_state import AppState
app_state = AppState()

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

        # create encoder
        self.encoder = EncoderRNN(input_voc_size=self.input_voc_size, hidden_size=self.hidden_size)

        # parse param to create decoder
        self.output_voc_size = params['output_voc_size']

        # create decoder
        self.decoder = DecoderRNN(hidden_size=self.hidden_size, output_voc_size=self.output_voc_size)

        # create attention decoder
        #self.decoder = AttnDecoderRNN(self.hidden_size, self.output_voc_size, dropout_p=0.1, max_length=self.max_length)

        print('Simple EncoderDecoderRNN (without attention) created.\n')

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

        # reshape tensors: from [batch_size x max_seq_length] to [max_seq_length x batch_size]
        input_tensor = inputs.transpose(0, 1)
        target_tensor = targets.transpose(0, 1)

        # init encoder hidden states
        encoder_hidden = self.encoder.init_hidden(batch_size)

        # for attention decoder !! Careful about the shape
        encoder_outputs = torch.zeros(self.max_length, batch_size, self.hidden_size).type(app_state.dtype)

        # encoder manual loop
        for ei in range(self.max_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei].unsqueeze(-1), encoder_hidden)
            encoder_outputs[ei] = encoder_output.squeeze()

        # TODO: check shape of encoder_outputs for attn_decoder
        encoder_outputs = encoder_outputs.transpose(0, 1)

        # decoder
        decoder_input = torch.ones(batch_size, 1).type(app_state.LongTensor) * SOS_token
        decoder_hidden = encoder_hidden

        decoder_outputs = torch.zeros(self.max_length, batch_size, self.output_voc_size).type(app_state.dtype)

        if self.training:  # Teacher forcing: Feed the target as the next input
            for di in range(self.max_length):
                # base decoder
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                # attention decoder
                #decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

                decoder_outputs[di] = decoder_output.squeeze()

                decoder_input = target_tensor[di].unsqueeze(-1)  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(self.max_length):
                #base decoder
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                # attention decoder
                #decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_outputs[di] = decoder_output.squeeze()

                topv, topi = decoder_output.topk(k=1, dim=-1)

                decoder_input = topi.view(batch_size, 1).detach()  # detach from history as input

                # TODO: fix this??
                #if decoder_input.item() == EOS_token:
                #    break

        return decoder_outputs.transpose(0, 1)


if __name__ == '__main__':
    # import lines for problem class
    import sys
    import os
    # add problem folder to path for import
    sys.path.insert(0, os.path.normpath(os.path.join(os.getcwd(), '../../problems/seq_to_seq/text2text')))
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

    params = {'batch_size': 1, 'training_size': 0.90, 'output_lang_name': 'fra', 'max_sequence_length': 10,
              'eng_prefixes': eng_prefixes, 'use_train_data': True, 'data_folder': '~/data/language', 'reverse': True}

    problem = pb.Translation(params)
    print('Problem successfully created.\n')

    # get size of vocabulary for input & output language
    input_voc_size = problem.input_lang.n_words
    output_voc_size = problem.output_lang.n_words

    # instantiate model with credible parameters
    model_params = {'max_length': 10, 'input_voc_size': input_voc_size, 'hidden_size': 256,
                    'output_voc_size': output_voc_size}
    net = SimpleEncoderDecoder(model_params)

    # generate a batch
    DataTuple, AuxTuple = problem.generate_batch()
    outputs = net(data_tuple=DataTuple)
    print('outputs: \n', outputs, '\n')

    # try to evaluate loss
    loss = problem.evaluate_loss(data_tuple=DataTuple, logits=outputs, aux_tuple=AuxTuple)
    print('loss: ', loss.item())
    loss.backward()
