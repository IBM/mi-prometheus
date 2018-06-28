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
        #self.decoder = DecoderRNN(hidden_size=self.hidden_size, output_voc_size=self.output_voc_size)

        # create attention decoder
        self.decoder = AttnDecoderRNN(self.hidden_size, self.output_voc_size, dropout_p=0.1, max_length=self.max_length)

        print('Simple EncoderDecoderRNN (without attention) created.\n')

    def reshape_tensor(self, tensor):
        """
        Helper function to reshape the tensor. Also removes padding (with PAD_token). E.g.
        tensor([[ 127, 223, 641, 5, 2, 0, 0, 0, 0, 0]]) -> return_tensor:  tensor([[ 127],
                                                                                   [ 223],
                                                                                   [ 641],
                                                                                   [   5],
                                                                                   [   2]])

        :param tensor: tensor to be reshaped & unpadded
        :return: transformed tensor.
        """
        # get indexes of elements not equal to ones
        index = (tensor[0, :] != PAD_token).nonzero().squeeze().numpy()
        # create new tensor being transposed compared to tensor + 1 element longer
        return_tensor = torch.ones(index.shape[0], 1).type(torch.long)
        # copy elements
        return_tensor[index] = tensor[0, index].view(-1, 1)

        return return_tensor

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

        # get batch_size, essentially equal to 1 for now
        batch_size = inputs.size(0)

        # reshape tensors
        # TODO: will need to be avoided in the future
        input_tensor = self.reshape_tensor(inputs)
        target_tensor = self.reshape_tensor(targets)

        # get sequences length
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        # init encoder hidden states
        encoder_hidden = self.encoder.init_hidden(batch_size)

        # for attention decoder
        encoder_outputs = torch.zeros(self.max_length, self.hidden_size)

        # encoder manual loop
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        # decoder
        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = encoder_hidden
        decoder_outputs = torch.zeros(target_length, self.output_voc_size)

        if self.training:  # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                # base decoder
                #decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                # attention decoder
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_outputs[di, :] = decoder_output.squeeze()

                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                #base decoder
                #decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                # attention decoder
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_outputs[di, :] = decoder_output.squeeze()

                topv, topi = decoder_output.topk(k=1, dim=-1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                if decoder_input.item() == EOS_token:
                    break

        return decoder_outputs


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
