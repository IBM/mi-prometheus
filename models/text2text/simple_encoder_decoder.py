#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of an Encoder-Decoder network for text2text problems (e.g. translation)"""
__author__ = "Vincent Marois "

import torch
from torch import nn
import torch.nn.functional as F
from misc.app_state import AppState

app_state = AppState()

SOS_token = 0
EOS_token = 1

class EncoderRNN(nn.Module):
    """GRU Encoder for Encoder-Decoder"""

    def __init__(self, input_voc_size, hidden_size):
        """
        Initializes an Encoder network based on a Gated Recurrent Unit.
        :param input_voc_size: size of the vocabulary set to be embedded by the Embedding layer.
        :param hidden_size: length of embedding vectors.
        """
        # call base constructor.
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size

        # Embedding: creates a look-up table of the embedding of a vocabulary set
        # (size: input_voc_size -> input_language.n_words) on vectors of size hidden_size.
        # adds 1 dimension to the shape of the tensor
        # WARNING: input must be of type LongTensor
        self.embedding = nn.Embedding(num_embeddings=input_voc_size, embedding_dim=hidden_size)

        # Apply a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
        # NOTE: default number of recurrent layers is 1
        # 1st parameter: expected number of features in the input -> same as hidden_size because of embedding
        # 2nd parameter: expected number of features in hidden state -> hidden_size.
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=False)

    def forward(self, input, hidden):
        """
        Runs the Encoder.
        :param input: tensor of indices, of size [batch_size x 1] (word by word looping)
        :param hidden: initial hidden state for each element in the input batch.
        Should be of size [1 x batch_size x hidden_size]
        :return: For every input word, the encoder outputs a vector and a hidden state,
                and uses the hidden state for the next input word.
                - output should be of size [batch_size x seq_len x hidden_size]: tensor containing the output
                features h_t from the last layer of the RNN, for each t.
                - hidden should be of size [1 x batch_size x hidden_size]: tensor containing the hidden state for
                t = seq_length
        """
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


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
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=False)

        # Apply a linear transformation to the incoming data: y=Ax+b
        # basically project from the hidden space to the output vocabulary set
        self.out = nn.Linear(in_features=hidden_size, out_features=output_voc_size)

        # Apply the Log(Softmax(x)) function to an n-dimensional input Tensor along the specified dimension
        # doesn't change the shape
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        """
        Runs the Decoder.
        :param input: tensor of indices, of size [batch_size x 1] (word by word looping)
        :param hidden: initial hidden state for each element in the input batch.
        Should be of size [1 x batch_size x hidden_size]
        :return:
                - output should be of size [batch_size x seq_len x output_voc_size] (unsqueezed): tensor containing the
                output features h_t from the last layer of the RNN, for each t.
                - hidden should be of size [1 x batch_size x hidden_size]: tensor containing the hidden state for
                t = seq_length
        """
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


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
        self.encoder = EncoderRNN(input_voc_size=self.input_voc_size, hidden_size=self.hidden_size)

        # parse param to create decoder
        self.output_voc_size = params['output_voc_size']

        # create decoder
        self.decoder = DecoderRNN(hidden_size=self.hidden_size, output_voc_size=self.output_voc_size)

        print('Simple EncoderDecoderRNN (without attention) created.\n')

    def reshape_tensor(self, tensor):
        """
        Helper function to reshape the tensor. Also removes padding (with 1s) except for the last element. E.g.
        tensor([[ 127,  223,  641,    5,    1,    1,    1,    1,    1,    1]]) -> return_tensor:  tensor([[ 127],
                                                                                                          [ 223],
                                                                                                          [ 641],
                                                                                                          [   5],
                                                                                                          [   1]])

        :param tensor: tensor to be reshaped & unpadded
        :return: transformed tensor.
        """
        # get indexes of elements not equal to ones
        index = (tensor[0, :] != 1).nonzero().squeeze().numpy()
        # create new tensor being transposed compared to tensor + 1 element longer
        return_tensor = torch.ones(index.shape[0] + 1, 1).type(torch.long)
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

        # encoder manual loop
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)

        # decoder
        decoder_input = torch.tensor([SOS_token])
        decoder_hidden = encoder_hidden
        decoder_outputs = torch.zeros(target_length, self.output_voc_size)

        if self.training:  # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                decoder_outputs[di, :] = decoder_output.squeeze()

                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
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
