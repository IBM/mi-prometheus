#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of an Encoder-Decoder network for text2text problems (e.g. translation)"""
__author__ = "Vincent Marois "

import torch
from torch import nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):

    """GRU Encoder for Encoder-Decoder"""
    def __init__(self, input_voc_size, hidden_size):
        """
        Initializes an Encoder network based on a Gated Recurrent Unit.
        :param input_voc_size: size of the vocabulary set to be embedded by the Embedding layer.
        :param hidden_size: length of embedding vectors.
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # Embedding: creates a look-up table of the embedding of a vocabulary set
        # (size: input_voc_size -> input_language.n_words) on vectors of size hidden_size.
        # adds 1 dimension to the shape of the tensor
        # WARNING: input must be of type LongTensor
        self.embedding = nn.Embedding(num_embeddings=input_voc_size, embedding_dim=self.hidden_size)

        # Apply a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
        # NOTE: default number of recurrent layers is 1
        # 1st parameter: expected number of features in the input -> same as hidden_size because of embedding
        # 2nd parameter: expected number of features in hidden state -> hidden_size.
        # batch_first=True -> input and output tensors are provided as [batch_size x seq_length x input_voc_size]
        # ! batch_first=True doesn't affect the hidden tensors
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)

    # encoder forward pass
    def forward(self, inputs, hidden):
        """
        Runs the Encoder.
        :param inputs: tensor of indices, of size [batch_size x sequence_length]
        :param hidden: initial hidden state for each element in the input batch.
        Should be of size [1 x batch_size x hidden_size]
        :return: For every input word, the encoder outputs a vector and a hidden state,
                and uses the hidden state for the next input word.
                - output should be of size [batch_size x seq_len x hidden_size]: tensor containing the output
                features h_t from the last layer of the RNN, for each t.
                - hidden should be of size [1 x batch_size x hidden_size]: tensor containing the hidden state for
                t = seq_length
        """
        embedded = self.embedding(inputs)
        # embedded shape = [batch_size x seq_length x hidden_size]
        output = embedded

        output, hidden = self.gru(output, hidden)

        return output, hidden

    def init_hidden(self, batch_size, dtype):
        """Returns initial hidden states set to 0.
            The shape is [1 x batch_size x hidden_size] to match with the GRU layer in the Encoder."""
        h_init = torch.zeros(1, batch_size, self.hidden_size).type(dtype)
        return h_init


class DecoderRNN(nn.Module):

    """GRU Decoder for Encoder-Decoder"""

    def __init__(self, hidden_size, output_voc_size):
        """
        Initializes an Decoder network based on a Gated Recurrent Unit.
        :param hidden_size: length of embedding vectors.
        :param output_voc_size: size of the vocabulary set to be embedded by the Embedding layer.
        """

        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # Embedding: creates a look-up table of the embedding of a vocabulary set
        # (size: output_voc_size -> input_language.n_words) on vectors of size hidden_size.
        # adds 1 dimension to the shape of the tensor
        # WARNING: input must be of type LongTensor
        self.embedding = nn.Embedding(num_embeddings=output_voc_size, embedding_dim=hidden_size)

        # Apply a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
        # NOTE: default number of recurrent layers is 1
        # 1st parameter: expected number of features in the input -> same as hidden_size because of embedding
        # 2nd parameter: expected number of features in hidden state -> hidden_size.
        # batch_first=True -> input and output tensors are provided as [batch_size x seq_length x input_voc_size]
        # ! batch_first=True doesn't affect the hidden tensors
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

        # Apply a linear transformation to the incoming data: y=Ax+b
        # basically project from the hidden space to the output vocabulary set
        self.out = nn.Linear(in_features=hidden_size, out_features=output_voc_size)

        # Apply the Log(Softmax(x)) function to an n-dimensional input Tensor along the specified dimension
        # doesn't change the shape
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, inputs, hidden):
        """
        Runs the Decoder.
        :param inputs: tensor of indices, of size [batch_size x sequence_length]
        :param hidden: initial hidden state for each element in the input batch.
        Should be of size [1 x batch_size x hidden_size]
        :return:
                - output should be of size [batch_size x seq_len x output_voc_size]: tensor containing the output
                features h_t from the last layer of the RNN, for each t.
                - hidden should be of size [1 x batch_size x hidden_size]: tensor containing the hidden state for
                t = seq_length
        """
        output = self.embedding(inputs.type(torch.long))
        # output should be of shape [batch_size x seq_length x hidden_size]

        output = F.relu(output)  # relu doesn't change the output shape

        output, hidden = self.gru(output, hidden)
        # output should be of shape [batch_size x seq_len x hidden_size]

        output = self.out(output)
        # output should be of shape [batch_size x seq_len x output_voc_size]

        # apply LogSoftmax along dimension 2 -> output_voc_size
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self, batch_size, dtype):
        """Returns initial hidden states set to 0.
        The shape is [1 x batch_size x hidden_size] to match with the GRU layer in the Decoder."""
        h_init = torch.zeros(1, batch_size, self.hidden_size).type(dtype)
        return h_init


class SimpleEncoderDecoder(nn.Module):
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
        super(SimpleEncoderDecoder, self).__init__()

        self.max_length = params['max_length']

        # parse params to create encoder
        self.input_voc_size = params['input_voc_size']
        self.hidden_size = params['hidden_size']

        # create encoder
        self.encoder = EncoderRNN(self.input_voc_size, self.hidden_size)

        # parse param to create decoder
        self.output_voc_size = params['output_voc_size']

        # create decoder
        self.decoder = DecoderRNN(self.hidden_size, self.output_voc_size)

        print('Simple EncoderDecoderRNN (without attention) created.\n')

    # global forward pass
    def forward(self, data_tuple):
        """
        Runs the network.
        :param data_tuple: (input_tensor, target_tensor) tuple
        :return: decoder output TODO: SPECIFY
        """
        # unpack data_tuple
        (inputs, targets) = data_tuple

        batch_size = inputs.size(0)

        # Check if the class has been converted to cuda (through .cuda() method)
        dtype = torch.cuda.FloatTensor if next(self.encoder.parameters()).is_cuda else torch.FloatTensor

        # initialize encoder hidden states to 0
        encoder_hidden = self.encoder.init_hidden(batch_size=batch_size, dtype=dtype)

        # encoder
        encoder_output, encoder_hidden = self.encoder(inputs=inputs, hidden=encoder_hidden)

        # decoder
        decoder_hidden = encoder_hidden

        if self.training:
            # teacher_forcing: feed the target as the next input -> equivalent to training
            decoder_input = targets
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

        else:  # without teacher_forcing: use its own predictions as the next input -> equivalent to inference
            decoder_input = torch.zeros_like(targets).type(dtype)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

        return decoder_output


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

    params = {'batch_size': 5, 'start_index': 0, 'stop_index': 1000, 'output_lang_name': 'fra',
              'max_sequence_length': 10, 'eng_prefixes': eng_prefixes, 'use_train_data': True,
              'data_folder': '~/data/language', 'reverse': False}

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

    # try to compute the BlEU score
    _, logits = outputs.topk(k=1, dim=2)
    bleu_score = problem.compute_BLEU_score(data_tuple=DataTuple, logits=logits.squeeze(), aux_tuple=AuxTuple,
                                            output_lang=problem.output_lang)
    print('BLEU score: ', bleu_score)
