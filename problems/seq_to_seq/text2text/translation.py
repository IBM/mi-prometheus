#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""translation.py: translation problem"""
__author__      = "Vincent Marois"

# Add path to main project directory - required for testing of the main function and see whether problem is working at all (!)
import os, sys
import random
import torch
import errno

sys.path.append(os.path.join(os.path.dirname(__file__),  '..', '..', '..'))

from problems.problem import DataTuple
from problems.seq_to_seq.text2text.text_to_text_problem import TextToTextProblem, Lang, TextAuxTuple


class Translation(TextToTextProblem, Lang):
    """
    Class generating sequences of indexes as inputs & targets for a English -> Other Language translation task.
    Only supports latin alphabet for now.
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

        # whether to reverse I/O languages or not
        self.reverse = params['reverse']

        # name the output language (input language is forced to English for now because of the data source)
        self.output_lang_name = params['output_lang_name']

        # max sequence length -> corresponds to max number of words in sentence
        self.max_sequence_length = params['max_sequence_length']

        # to filter the English sentences based on their structure.
        self.eng_prefixes = params['eng_prefixes']

        self.start_index = params['start_index']
        self.stop_index = params['stop_index']

        # other attributes
        self.input_lang = None  # will be a Lang instance
        self.output_lang = None  # will be a Lang instance
        self.pairs = []  # will be used to constitute TextAuxTuple
        self.tensor_pairs = []  # will be used to constitute DataTuple

        # for datasets storage & handling
        self.root = os.path.expanduser(params['data_folder'])
        self.raw_folder = 'raw'
        self.processed_folder = 'processed'
        self.training_file = 'eng-' + self.output_lang_name + '_training.txt'
        self.test_file = 'eng-' + self.output_lang_name + '_test.txt'
        self.training_size = 0.90

        # switch between training & inference
        self.use_train_data = params['use_train_data']

        # create corresponding Lang instances using the names
        self.input_lang = Lang('eng')
        self.output_lang = Lang(self.output_lang_name)

        # preprocess source data
        self.download()
        self.input_lang, self.output_lang, self.pairs = self.prepare_data()

        # create tensors of indexes from string pairs
        self.tensor_pairs = self.tensors_from_pairs(self.pairs, self.input_lang,
                                                    self.output_lang, self.max_sequence_length)

        # number of training instances
        if self.use_train_data:
            assert self.stop_index < len(self.pairs), "Error: specified stop_index > number of processed training pairs."
            self.num_train = int(self.stop_index - self.start_index)

    def prepare_data(self):
        """
        Prepare the data for generating batches. Uses read_langs() & filter_pairs() to normalize, trim & filter input
        sentences pairs.
        Also fills in Lang() instances for the input & output languages.
        :return: Lang() object for input & output languages + filtered sentences pairs.
        """

        # Read the source data file and split into lines
        if self.use_train_data:
            print('Using training set')
            lines = open(os.path.join(self.root, self.processed_folder, self.training_file), encoding='utf-8').\
                read().strip().split('\n')
        else:
            print('Using inference set')
            lines = open(os.path.join(self.root, self.processed_folder, self.test_file), encoding='utf-8').\
                read().strip().split('\n')

        # Split every line into pairs and normalize them
        self.pairs = [[self.normalize_string(s) for s in l.split('\t')] for l in lines]

        print("Read %s sentence pairs" % len(self.pairs))

        # filter sentences pairs (based on number of words & prefixes).
        self.pairs = self.filter_pairs()

        # if reverse, switch input & output sentences.
        if self.reverse:
            self.pairs = [list(reversed(p)) for p in self.pairs]
            self.input_lang = Lang(self.output_lang_name)
            self.output_lang = Lang('eng')

        print("Trimmed to %s sentence pairs" % len(self.pairs))

        # fill in Lang() objects with some info
        for pair in self.pairs:
            self.input_lang.add_sentence(pair[0])
            self.output_lang.add_sentence(pair[1])
        print("Number of words in I/O languages:")
        print(self.input_lang.name, ':', self.input_lang.n_words)
        print(self.output_lang.name, ':', self.output_lang.n_words)

        return self.input_lang, self.output_lang, self.pairs

    def _check_exists(self):
        """Check if the training & inference datasets for the specified language already exist or not."""
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """
        Download the specified zip file from http://www.manythings.org/anki/.
        Notes: This website hosts data files for English -> other language translation: the main file is named after
        the other language.
            Ex: for a English -> French translation, the main file is named 'fra.txt',
                for a English -> German translation, the main file is named 'deu.txt' etc.
        """
        # import lines
        from six.moves.urllib.request import Request, urlopen
        import zipfile

        # check if the files already exist
        if self._check_exists():
            print('Files already exist, no need to re-download them.')
            return

        # try to create directories for storing files if not already exist
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # construct the url from self.output_lang_name
        # Warning: The source files are named like 'eng-fra.zip' -> careful on the language abbreviation!
        url = 'http://www.manythings.org/anki/' + self.output_lang_name + '-eng.zip'

        print('Downloading original source file from', url)
        # have to do a Request in order to pass headers to avoid server security features blocking spider/bot user agent
        request = Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11'})
        data = urlopen(request)

        # write raw data to file
        filename = url.rpartition('/')[2]
        filepath = os.path.join(self.root, self.raw_folder, filename)

        with open(filepath, 'wb') as f:
            f.write(data.read())
        with zipfile.ZipFile(filepath, 'r') as zip_f:
            zip_f.extractall(os.path.join(self.root, self.raw_folder))
        os.unlink(filepath)

        # read raw data, split it in training & inference sets and save it to file
        lines = open(os.path.join(self.root, self.raw_folder, self.output_lang_name + '.txt'), encoding='utf-8').\
            read().strip().split('\n')

        nb_samples = len(lines)
        print('Total number of samples:', nb_samples)
        nb_training_samples = round(self.training_size * nb_samples)
        training_samples = lines[:nb_training_samples]
        inference_samples = lines[nb_training_samples:]

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'w') as training_f:
            training_f.write('\n'.join(line for line in training_samples))
            training_f.close()
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'w') as test_f:
            test_f.write('\n'.join(line for line in inference_samples))
            test_f.close()

        print('Processing done.')

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
               p[0].startswith(self.eng_prefixes)

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
        # only considering the start / stop indexes for the training set
        if self.use_train_data:
            indexes = random.sample(population=range(self.num_train), k=self.batch_size)
        else:
            indexes = random.sample(population=range(len(self.tensor_pairs)), k=self.batch_size)

        # create main batch inputs & outputs tensor
        inputs = torch.zeros([self.batch_size, self.max_sequence_length], dtype=torch.long)
        targets = torch.zeros([self.batch_size, self.max_sequence_length], dtype=torch.long)

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
        aux_tuple = TextAuxTuple(inputs_text, outputs_text, self.input_lang, self.output_lang)

        return data_tuple, aux_tuple


if __name__ == "__main__":
    """ Tests Problem class"""

    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )

    params = {'batch_size': 2, 'start_index': 0, 'stop_index': 1000, 'output_lang_name': 'fra',
              'max_sequence_length': 10, 'eng_prefixes': eng_prefixes, 'use_train_data': True,
              'data_folder': '~/data/language', 'reverse': False}

    problem = Translation(params)
    print('Problem successfully created.\n')

    generator = problem.return_generator()
    # Get batch.
    data_tuple, aux_tuple = next(generator)

    print('data_tuple: ', data_tuple)
    print('aux_tuple: ', aux_tuple)
