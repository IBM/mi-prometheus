#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""translation.py: translation problem"""
__author__      = "Vincent Marois"

# Add path to main project directory - required for testing of the main function and see whether problem is working at all (!)
import os, sys
import random
import torch
sys.path.append(os.path.join(os.path.dirname(__file__),  '..', '..', '..'))

from torch.utils.data.sampler import SubsetRandomSampler
from problems.problem import DataTuple
from problems.seq_to_seq.text2text.text_to_text_problem import TextToTextProblem, Lang, TextAuxTuple


class Translation(TextToTextProblem, Lang):
    """
    Class generating sequences of indexes as inputs & targets for a translation task.
    TODO: padding for sequences of different lengths in batch needs to be checked
    """

    def __init__(self, params):
        """
        Initializes the problem: stores parameters. Calls parent class initialization.
        :param params: Dictionary of parameters.
        """

        # Call parent constructor - e.g. sets the default loss function
        super(Translation, self).__init__(params)

        # parse parameters from the dictionary.
        self.batch_size = params['batch_size']

        # source data filepath TODO: in future, should manage automatic download & storage.
        self.source_data_filepath = params['source_data_filepath']
        assert self.source_data_filepath != '', 'The source data filepath cannot be empty.'

        # simple strings to name the input & output languages
        self.input_lang_name = params['input_lang_name']
        self.output_lang_name = params['output_lang_name']

        # max sequence length -> corresponds to max number of words in sentence
        self.max_sequence_length = params['max_sequence_length']

        # to filter the input sentences based on their structure.
        self.input_lang_prefixes = params['input_lang_prefixes']

        # TODO: is it useful? How to delimitate train & test dataset?
        self.start_index = params['start_index']
        self.stop_index = params['stop_index']

        # other attributes
        self.input_lang = None  # will be a Lang instance
        self.output_lang = None  # will be a Lang instance
        self.pairs = []  # will be used to constitute TextAuxTuple
        self.tensor_pairs = []  # will be used to constitute DataTuple
        self.gpu = False  # TODO: Problem will need to be prepared for CUDA

        # create corresponding Lang instances using the names
        self.input_lang = Lang(self.input_lang_name)
        self.output_lang = Lang(self.output_lang_name)

        # preprocess source data
        self.input_lang, self.output_lang, self.pairs = self.prepare_data()

        # create tensors of indexes from string pairs
        self.tensor_pairs = self.tensors_from_pairs(self.pairs, self.input_lang,
                                                    self.output_lang, self.max_sequence_length)

        # number of training instances
        assert self.stop_index < len(self.pairs), "Error: specified stop_index > number of processed pairs."
        self.num_train = int(self.stop_index - self.start_index)

    def prepare_data(self):
        """
        Prepare the data for generating batches. Uses read_langs() & filter_pairs() to normalize, trim & filter input
        sentences pairs.
        Also fills in Lang() instances for the input & output languages.
        :return: Lang() object for input & output languages + filtered sentences pairs.
        """

        # Read the source data file and split into lines
        lines = open(self.source_data_filepath, encoding='utf-8').read().strip().split('\n')

        # Split every line into pairs and normalize them
        self.pairs = [[self.normalize_string(s) for s in l.split('\t')] for l in lines]

        print("Read %s sentence pairs" % len(self.pairs))

        # filter sentences pairs (based on number of words & prefixes).
        self.pairs = self.filter_pairs()

        print("Trimmed to %s sentence pairs" % len(self.pairs))

        # fill in Lang() objects with some info
        for pair in self.pairs:
            self.input_lang.add_sentence(pair[0])
            self.output_lang.add_sentence(pair[1])
        print("Number of words in I/O languages:")
        print(self.input_lang.name, ':', self.input_lang.n_words)
        print(self.output_lang.name, ':', self.output_lang.n_words)

        return self.input_lang, self.output_lang, self.pairs

    def filter_pair(self, p):
        """
        Indicate whether a sentence pair is compliant with some filtering criteria, such as:
         - The number of words (that includes ending punctuation) in the sentences,
         - The start of the input language sentence.

        :param p: [] containing a pair of sentences

        :return: True if the pair respects the filtering constraints else False.
        """

        return len(p[0].split(' ')) < self.max_sequence_length and \
               len(p[1].split(' ')) < self.max_sequence_length and \
               p[0].startswith(self.input_lang_prefixes)

    def filter_pairs(self):
        """Filter several pairs at once using filter_pair as a boolean mask.
        :return list of filtered pairs"""
        return [pair for pair in self.pairs if self.filter_pair(pair)]

    def generate_batch(self):
        """
        Generates a batch  of size [BATCH_SIZE, MAX_SEQUENCE_LENGTH, 1].

        :return: Tuple consisting of: input [BATCH_SIZE, MAX_SEQUENCE_LENGTH, 1],
                                    output [BATCH_SIZE, MAX_SEQUENCE_LENGTH, 1].
        """
        # generate a sample of size batch_size of random indexes without replacement
        indexes = random.sample(population=range(self.num_train), k=self.batch_size)

        # create main batch inputs & outputs tensor
        inputs = torch.zeros([self.batch_size, self.max_sequence_length])
        targets = torch.zeros([self.batch_size, self.max_sequence_length])

        # for TextAuxTuple
        inputs_text = []
        outputs_text = []

        for i, index in enumerate(indexes):
            input_tensor, target_tensor = self.tensor_pairs[index]
            input_text, output_text = self.pairs[index]

            inputs[i] = input_tensor
            targets[i] = target_tensor
            inputs_text.append(input_text)
            outputs_text.append(output_text)

        # Return tuples.
        data_tuple = DataTuple(inputs, targets)
        aux_tuple = TextAuxTuple(0, inputs_text, outputs_text)

        return data_tuple, aux_tuple


if __name__ == "__main__":
    """ Tests Problem class"""

    input_lang_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )

    params = {'batch_size': 2, 'start_index': 0, 'stop_index': 1000, 'source_data_filepath': 'eng-fra.txt',
              'input_lang_name': 'english', 'output_lang_name': 'french', 'max_sequence_length': 10,
              'input_lang_prefixes': input_lang_prefixes}

    problem = Translation(params)
    print('Problem successfully created.\n')

    generator = problem.return_generator()
    # Get batch.
    data_tuple, aux_tuple = next(generator)

    print('data_tuple: ', data_tuple)
    print('aux_tuple: ', aux_tuple)
