#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# The MIT License (MIT)
#
# Copyright (c) 2017 Sean Robertson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
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
translation.py: 'toy' translation problem class

"""
__author__ = "Vincent Marois"

import os
import random

import torch
import errno

from problems.problem import DataDict
from problems.seq_to_seq.text2text.text_to_text_problem import TextToTextProblem, Lang


class TranslationAnki(TextToTextProblem):
    """
    Class generating sequences of indexes as inputs & targets for a English <->
    Other Language translation task.

    .. warning::

        The inspiration for this class being an existing PyTorch tutorial, this class is limited.

        It currently only supports the files located here: http://www.manythings.org/anki/

        It currently only supports latin alphabet for now (because of string normalization) and does not
        include advanced features like beam search or pretrained embeddings.

        Take this class as an example and not as a production-ready application.


    """

    def __init__(self, params):
        """
        Initializes the problem: stores parameters. Calls parent class ``TextToTextProblem``
        initialization.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).

        """
        # Call parent constructor - e.g. sets the default loss function
        super(TranslationAnki, self).__init__(params)

        # whether to reverse I/O languages or not
        self.reverse = params['reverse']

        # name the output language
        self.output_lang_name = params['output_lang_name']

        # max sequence length -> corresponds to max number of words in sentence
        self.max_sequence_length = params['max_sequence_length']

        # to filter the English sentences based on their structure.
        self.eng_prefixes = params['eng_prefixes']

        # other attributes
        self.input_lang = None  # will be a Lang instance
        self.output_lang = None  # will be a Lang instance
        self.pairs = []  # will be used to constitute TextAuxTuple
        self.tensor_pairs = []  # will be used to constitute DataTuple

        # for datasets storage & handling
        self.root = os.path.expanduser(params['data_folder'])
        self.raw_folder = 'raw'
        self.processed_folder = 'processed'
        self.training_size = params['training_size']

        self.training_file = 'eng-' + self.output_lang_name + '_training_' + str(self.training_size) + '.txt'
        self.test_file = 'eng-' + self.output_lang_name + '_test_' + str(self.training_size) + '.txt'

        # switch between training & inference datasets
        self.use_train_data = params['use_train_data']

        # create corresponding Lang instances using the names
        self.input_lang = Lang('eng')
        self.output_lang = Lang(self.output_lang_name)

        # preprocess source data
        self.download()
        self.input_lang, self.output_lang, self.pairs = self.prepare_data()

        # create tensors of indexes from string pairs
        self.tensor_pairs = self.tensors_from_pairs(
            self.pairs, self.input_lang, self.output_lang, self.max_sequence_length)

        self.length = len(self.tensor_pairs)

    def prepare_data(self):
        """
        Prepare the data for generating batches.

        Uses ``filter_pairs()`` to normalize, trim & filter input sentences pairs.
        Also fills in ``Lang()`` instances for the input & output languages.

        :return: ``Lang()`` object for input & output languages + filtered sentences pairs.

        """

        # Read the source data file and split into lines
        if self.use_train_data:
            self.logger.info('Using training set')
            lines = open(
                os.path.join(
                    self.root,
                    self.processed_folder,
                    self.training_file),
                encoding='utf-8'). read().strip().split('\n')
        else:
            self.logger.info('Using inference set')
            lines = open(
                os.path.join(
                    self.root,
                    self.processed_folder,
                    self.test_file),
                encoding='utf-8'). read().strip().split('\n')

        # Split every line into pairs and normalize them
        self.pairs = [[self.normalize_string(s)
                       for s in l.split('\t')] for l in lines]

        self.logger.info("Read {} sentence pairs".format(len(self.pairs)))

        # shuffle pairs of sentences
        random.shuffle(self.pairs)

        # filter sentences pairs (based on number of words & prefixes).
        self.pairs = self.filter_pairs()

        # if reverse, switch input & output sentences.
        if self.reverse:
            self.pairs = [list(reversed(p)) for p in self.pairs]
            self.input_lang = Lang(self.output_lang_name)
            self.output_lang = Lang('eng')

        self.logger.info("Trimmed to {} sentence pairs".format(len(self.pairs)))

        # fill in Lang() objects with some info
        for pair in self.pairs:
            self.input_lang.add_sentence(pair[0])
            self.output_lang.add_sentence(pair[1])
        self.logger.info("Number of words in I/O languages:")
        self.logger.info('{}: {}'.format(self.input_lang.name, self.input_lang.n_words))
        self.logger.info('{}: {}'.format(self.output_lang.name, self.output_lang.n_words))

        return self.input_lang, self.output_lang, self.pairs

    def _check_exists(self):
        """
        :return: True if the training & inference datasets (of the specified training\
         size) for the specified language already exist or not.

        """
        return os.path.exists(
            os.path.join(
                self.root,
                self.processed_folder,
                self.training_file)) and os.path.exists(
            os.path.join(
                self.root,
                self.processed_folder,
                self.test_file))

    def download(self):
        """
        Download the specified zip file from http://www.manythings.org/anki/.
        Notes: This website hosts data files for English -> other language translation: the main file is named after
        the other language.

        Ex: for a English -> French translation, the main file is named 'fra.txt',

        Ex: for a English -> German translation, the main file is named 'deu.txt' etc.

        """
        # import lines
        from six.moves.urllib.request import Request, urlopen
        import zipfile

        # check if the files already exist
        if self._check_exists():
            self.logger.warning('Files already exist, no need to re-download them.')
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
        # Warning: The source files are named like 'eng-fra.zip' -> careful on
        # the language abbreviation!
        url = 'http://www.manythings.org/anki/' + self.output_lang_name + '-eng.zip'

        self.logger.warning('Downloading original source file from {}'.format(url))
        # have to do a Request in order to pass headers to avoid server
        # security features blocking spider/bot user agent
        request = Request(
            url, headers={
                'User-Agent': 'Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11'})
        data = urlopen(request)

        # write raw data to file
        filename = url.rpartition('/')[2]
        filepath = os.path.join(self.root, self.raw_folder, filename)

        with open(filepath, 'wb') as f:
            f.write(data.read())
        with zipfile.ZipFile(filepath, 'r') as zip_f:
            zip_f.extractall(os.path.join(self.root, self.raw_folder))
        os.unlink(filepath)

        # read raw data, split it in training & inference sets and save it to
        # file
        lines = open(
            os.path.join(
                self.root,
                self.raw_folder,
                self.output_lang_name +
                '.txt'),
            encoding='utf-8'). read().strip().split('\n')

        # shuffle list of lines
        random.shuffle(lines)

        nb_samples = len(lines)
        self.logger.info('Total number of samples: {}'.format(nb_samples))
        nb_training_samples = round(self.training_size * nb_samples)

        # choose nb_training_samples elements at random in lines to create the
        # training set
        training_samples_index = random.sample(
            range(len(lines)), nb_training_samples)
        training_samples = []
        for index in sorted(training_samples_index, reverse=True):
            training_samples.append(lines.pop(index))
        inference_samples = lines

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'w') as training_f:
            training_f.write('\n'.join(line for line in training_samples))
            training_f.close()
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'w') as test_f:
            test_f.write('\n'.join(line for line in inference_samples))
            test_f.close()

        self.logger.info('Processing done.')

    def filter_pair(self, p):
        """
        Indicate whether a sentence pair is compliant with some filtering
        criteria, such as:

         - The number of words (that includes ending punctuation) in the sentences,
         - The start of the input language sentence.

        :param p: pair of sentences
        :type p: list

        :return: True if the pair respects the filtering constraints else False.

        """
        if self.eng_prefixes is not None:

            return len(p[0].split(' ')) < self.max_sequence_length and \
                len(p[1].split(' ')) < self.max_sequence_length and \
                p[0].startswith(tuple(self.eng_prefixes))
        else:  # if no english prefixes have been specified, only filter based on sequence length
            return len(p[0].split(' ')) < self.max_sequence_length and \
                len(p[1].split(' ')) < self.max_sequence_length

    def filter_pairs(self):
        """
        Filter several pairs at once using filter_pair as a boolean mask.

        :return list of filtered pairs.

        """
        return [pair for pair in self.pairs if self.filter_pair(pair)]

    def __getitem__(self, index):
        """
        Retrieves a sample from ``self.tensor_pairs`` and get the associated strings from ``self.pairs``.


        :param index: index of the sample to return.
        :type index: int

        :return: DataDict({'sequences', 'sequences_length', 'targets', 'mask', 'inputs_text', 'outputs_text'}).

        """
        # get tensors and strings
        input_tensor, target_tensor = self.tensor_pairs[index]
        input_text, target_text = self.pairs[index]

        # return data_dict
        data_dict = DataDict({key: None for key in self.data_definitions.keys()})
        data_dict['sequences'] = input_tensor
        data_dict['sequences_length'] = len(input_tensor)
        data_dict['mask'] = 0
        data_dict['targets'] = target_tensor
        data_dict['inputs_text'] = input_text
        data_dict['outputs_text'] = target_text

        return data_dict

    def collate_fn(self, batch):
        """
        Combines a list of ``DataDict`` (retrieved with ``__getitem__`` ) into a batch.

        .. note::

            This function wraps a call to ``default_collate`` and simply returns the batch as a ``DataDict``\
            instead of a dict.

        :param batch: list of individual ``DataDict`` samples to combine.

        :return: ``DataDict({'sequences', 'sequences_length', 'targets', 'mask', 'inputs_text', 'outputs_text'}).`` containing the batch.

        """
        return DataDict({key: value for key, value in zip(self.data_definitions.keys(),
                                                          super(TranslationAnki, self).collate_fn(batch).values())})c

    def plot_preprocessing(self, data_dict, logits):
        """
        Does some preprocessing to logits to then plot the attention weights
        for the AttnEncoderDecoder model.

        :param data_dict: DataDict({'sequences', 'sequences_length', 'targets', 'mask', 'inputs_text', 'outputs_text'}).

        :param logits: prediction, shape [batch_size x max_seq_length x output_voc_size]
        :return: data_dict, + logits as dict {'inputs_text', 'logits_text'}

        """
        # get most probable words indexes for the batch
        _, top_indexes = logits.topk(k=1, dim=-1)
        top_indexes = top_indexes.squeeze()

        # retrieve text sentences from the logits (which should be tensors of
        # indexes)
        logits_text = []
        for logit in top_indexes:
            logits_text.append(
                [self.output_lang.index2word[index.item()]
                 for index in logit])

        # cannot modify DataTuple so modifying logits to contain the input
        # sentences and predicted sentences
        logits = {'inputs_text': data_dict['inputs_text'],
                  'logits_text': logits_text}
o
        return data_dict, logits


if __name__ == "__main__":
    """
    Problem class Unit Test.
    """

    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )

    # Load parameters.
    from utils.param_interface import ParamInterface
    params = ParamInterface()
    params.add_default_params({'training_size': 0.9,
                               'output_lang_name': 'fra',
                               'max_sequence_length': 15,
                               'eng_prefixes': eng_prefixes,
                               'use_train_data': True,
                               'data_folder': '~/data/language',
                               'reverse': False})

    batch_size = 64

    # Create problem.
    translation = TranslationAnki(params)

    # get a sample
    sample = translation[10]
    print(repr(sample))
    print('__getitem__ works.')

    # wrap DataLoader on top of this Dataset subclass
    from torch.utils.data.dataloader import DataLoader
    dataloader = DataLoader(dataset=translation, collate_fn=translation.collate_fn,
                            batch_size=batch_size, shuffle=True, num_workers=0)

    # try to see if there is a speed up when generating batches w/ multiple workers
    import time
    s = time.time()
    for i, batch in enumerate(dataloader):
        print('Batch # {} - {}'.format(i, repr(batch)))
        break

    print('Number of workers: {}'.format(dataloader.num_workers))
    print('time taken to exhaust the dataset for a batch size of {}: {}s'.format(batch_size, time.time()-s))

    # Display single sample (0) from batch.
    #batch = next(iter(dataloader))
    #translation.show_sample(batch, 0)

    print('Unit test completed')