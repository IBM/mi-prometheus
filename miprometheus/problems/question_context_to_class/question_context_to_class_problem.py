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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, e:wqither express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -*- coding: utf-8 -*-

"""question_context_to_class_problem.py: contains abstract base class for context to class problems"""
__author__      = "Tomasz Kornuta, Vincent Albouy"

import collections
from problems.problem import Problem
from utils.loss.masked_cross_entropy_loss import MaskedCrossEntropyLoss
from tqdm import tqdm
from torch.utils.data import Dataset
import requests
import os
import zipfile
import tarfile
import gzip
import shutil
import torch.utils.data
from utils.app_state import AppState

_DataTuple = collections.namedtuple('DataTuple', ('inputs', 'targets'))

class DataTuple(_DataTuple):
    """Tuple used to store batches of question-context pairs e.g. QA problems"""
    __slots__ = ()

_AuxTuple = collections.namedtuple('AuxTuple', ('story','answer', 'mask'))

class AuxTuple(_AuxTuple):
    """Tuple used to store batches of question-context pairs e.g. QA problems"""
    __slots__ = ()

class QuestionContextToClass(Problem):
    ''' Abstract base class for QA  (Question Answering) problems. Provides some basic functionality usefull in all problems of such type'''

    def __init__(self, params):
        """ 
        Initializes problem, calls base class initialization. Set loss function to CrossEntropy.

        :param params: Dictionary of parameters (read from configuration file).        
        """ 
        # Call base class constructors.
        super(QuestionContextToClass, self).__init__(params)

        self.loss_function = MaskedCrossEntropyLoss()

    def calculate_accuracy(self, data_tuple, logits, aux_tuple):
        """ Calculates accuracy equal to mean number of correct answers in a given batch.
        WARNING: Applies mask (from aux_tuple) to logits!
        
        :param logits: Logits being output of the model.
        :param data_tuple: Data tuple containing inputs and targets.
        :param _: auxiliary tuple (aux_tuple) is not used in this function. 
        """
        # Unpack tuple.
        (_, targets) = data_tuple
        (_, _, mask) = aux_tuple
        
        pred = logits.transpose(1,2)
        return self.loss_function.masked_accuracy(pred, targets, mask)

    def print_results(self, pred, targets, data_tuple, aux_tuple):
        pass

    def evaluate_loss(self, data_tuple, logits, aux_tuple):
        """ Calculates loss between the predictions/logits and targets (from data_tuple) using the selected loss function.
        
        :param logits: Logits being output of the model.
        :param data_tuple: Data tuple containing inputs and targets.
        :param _: auxiliary tuple (aux_tuple) is not used in this function. 
        """
        # Unpack tuple.
        (_, targets) = data_tuple
        (_, _, mask) = aux_tuple
        pred = logits.transpose(1,2)
        loss = self.loss_function(pred, targets, mask)

        return loss


    def add_statistics(self, stat_col):
        """
        Add accuracy statistic to collector. 

        :param stat_col: Statistics collector.
        """
        stat_col.add_statistic('acc', '{:12.10f}')

    def collect_statistics(self, stat_col, data_tuple, logits, _):
        """
        Collects accuracy.

        :param stat_col: Statistics collector.
        :param data_tuple: Data tuple containing inputs and targets.
        :param logits: Logits being output of the model.
        :param _: auxiliary tuple (aux_tuple) is not used in this function. 
        """
        stat_col['acc'] = self.calculate_accuracy(data_tuple, logits, _)

    def turn_on_cuda(self, data_tuple, aux_tuple):
        """ Enables computations on GPU - copies the input and target matrices (from DataTuple) to GPU.
        This method has to be overwritten in derived class if one decides to copy other matrices as well.

        :param data_tuple: Data tuple.
        :param aux_tuple: Auxiliary tuple (WARNING: Values stored in that variable will remain in CPU)
        :returns: Pair of Data and Auxiliary tuples (Data on GPU, Aux on CPU).
        """
        # Unpack tuples and copy data to GPU.
        gpu_question = data_tuple.inputs.cuda()
        gpu_targets = data_tuple.targets.cuda()

        # Pack tensors into tuples
        data_tuple = DataTuple(gpu_question, gpu_targets)

        return data_tuple, aux_tuple


    def download(self, root, check=None):
        #REPLACE WITH OWN CODE
        """Download and unzip an online archive (.zip, .gz, or .tgz).
        Arguments:
            root (str): Folder to download data to.
            check (str or None): Folder whose existence indicates
                that the dataset has already been downloaded, or
                None to check the existence of root/{cls.name}.
        Returns:
            str: Path to extracted dataset.
        """
        path = os.path.join(root, self.name)
        check = path if check is None else check
        #print(check)

        if not os.path.isdir(check):
            #print("pass")
            for url in self.urls:
                if isinstance(url, tuple):
                    url, filename = url
                else:
                    filename = os.path.basename(url)
                zpath = os.path.join(path, filename)
                if not os.path.isfile(zpath):
                    if not os.path.exists(os.path.dirname(zpath)):
                        os.makedirs(os.path.dirname(zpath))
                    print('downloading {}'.format(filename))
                    self.download_from_url(url, zpath)
                zroot, ext = os.path.splitext(zpath)
                _, ext_inner = os.path.splitext(zroot)
                if ext == '.zip':
                    with zipfile.ZipFile(zpath, 'r') as zfile:
                        print('extracting')
                        zfile.extractall(path)
                # tarfile cannot handle bare .gz files
                elif ext == '.tgz' or ext == '.gz' and ext_inner == '.tar':
                    with tarfile.open(zpath, 'r:gz') as tar:
                        dirs = [member for member in tar.getmembers()]
                        tar.extractall(path=path, members=dirs)
                elif ext == '.gz':
                    with gzip.open(zpath, 'rb') as gz:
                        with open(zroot, 'wb') as uncompressed:
                            shutil.copyfileobj(gz, uncompressed)

        return os.path.join(path, self.dirname)

    def build_dictionaries_one_hot(self):

        tasks = self.tasks

        data = self.load_data(tasks=tasks, tenK=self.tenK, add_punctuation=True, data_type='train', outmod="one_hot")
        data = data + self.load_data(tasks=tasks, tenK=self.tenK, add_punctuation=True, data_type='valid',
                                     outmod="one_hot")
        data = data + self.load_data(tasks=tasks, tenK=self.tenK, add_punctuation=True, data_type='test',
                                     outmod="one_hot")

        answ_to_ix = {".": 0, "?": 1, "_": 2}
        itos_d = [".", "?", "_"]
        self.fix_length = 0
        for q in tqdm(data):
            # display a progress bar
            # question = self.tokenize(q['question'])
            # answer = q['answer']
            story, answers = q
            self.fix_length = max(self.fix_length, len(story))
            # print(self.fix_length, len(story))
            # if len(story) > 200:
            # print(story)
            # input()
            for answer in story:
                a = answer.lower()
                if a not in answ_to_ix:
                    ix = len(answ_to_ix)
                    answ_to_ix[a] = ix
                    itos_d.append(a)
                    # print(a,ix)

            for answer in answers:
                a = answer.lower()
                if a not in answ_to_ix:
                    ix = len(answ_to_ix)
                    answ_to_ix[a] = ix
                    itos_d.append(a)
                    # print(a, ix)

        ret = (answ_to_ix)
        return ret, itos_d

    def build_dictionaries(self):
        """Creates the word embeddings

        - 1. Collects all datasets word
        - 2. Uses Language object to create the embeddings

         If it is the first time you run this code, it will take longer to load the embedding from torchtext
         """

        print(' ---> Constructing the dictionaries with word embedding, may take some time ')

        # making an empty list of words meant to store all possible datasets words
        text = []
        tasks = self.tasks

        data = self.load_data(tasks=tasks, tenK=self.tenK, add_punctuation=True, data_type='train', outmod="embedding")
        data = data + self.load_data(tasks=tasks, tenK=self.tenK, add_punctuation=True, data_type='valid',
                                     outmod="embedding")
        data = data + self.load_data(tasks=tasks, tenK=self.tenK, add_punctuation=True, data_type='test',
                                     outmod="embedding")

        answ_to_ix = {".": 0, "?": 1, "_": 2}
        itos_d = [".", "?", "_"]
        # load all words from training data to a list named words_list[]
        self.fix_length = 0
        for q in tqdm(data):
            # display a progress bar
            story, answers = q
            self.fix_length = max(self.fix_length, len(story))
            for word in story:
                text.extend([word.lower()])

            for answer in story:
                a = answer.lower()
                if a not in answ_to_ix:
                    ix = len(answ_to_ix)
                    answ_to_ix[a] = ix
                    itos_d.append(a)
                    # print(a,ix)

            for answer in answers:
                a = answer.lower()
                if a not in answ_to_ix:
                    ix = len(answ_to_ix)
                    answ_to_ix[a] = ix
                    itos_d.append(a)
                    # print(a, ix)

        """ build embeddings from the chosen database / Example: glove.6B.100d """

        self.language.build_pretrained_vocab(text, vectors=self.embedding_type, tokenize=self.tokenize)

        ret = (answ_to_ix)
        return ret, itos_d

    def to_dictionary_indexes(self, dictionary, sentence):
        """
        Outputs indexes of the dictionary corresponding to the words in the sequence.
        Case insensitive.
        """

        idxs = torch.tensor([dictionary[w.lower()] for w in sentence]).type(AppState().LongTensor)
        return idxs

    def indices_to_words(self, int_sentence):

        sentences = []
        for ind in int_sentence[0, :]:
            sentences.append(self.itos_dict[ind])
        return sentences

    def embed_sentence_one_hot(self, sentence):
        """
        Embed an entire sentence using a pretrained embedding
        :param sentence: A string containing the words to embed
        :returns: FloatTensor of embedded vectors [max_sentence_length, embedding size]
        """
        size_hot = len(self.dictionaries)
        outsentence = torch.zeros((len(sentence.split(" ")), size_hot))
        # for key, value in self.dictionaries.items():
        #    print(key, value)

        # print(size_hot)
        # embed a word at a time
        for i, word in enumerate(sentence.split(" ")):
            if not word.lower() == self.pad_token:
                index = self.dictionaries[word.lower()]
                # print(index, word)
                outsentence[i, index] = 1
                # print(outsentence[i,:])

        return outsentence

        # Change name to embed sentence

    def embed_batch(self, minibatch):

        ex = minibatch
        sentence = " ".join(ex)

        if self.one_hot_embedding:
            sent_embed = self.embed_sentence_one_hot(sentence)
        else:
            sent_embed = self.language.embed_sentence(sentence)

        return sent_embed

    def tokenize(self, sentence):
        return sentence.split(' ')

        # list to string

    def detokenize_story(self, minibatch):
        a = []
        for ex in minibatch:
            b = []
            # print(ex)
            for sentence in ex:
                b.append(" ".join(sentence))
            a.append(b)
        return a

        # string to list

    def tokenize_story(self, minibatch):
        a = []
        for ex in minibatch:
            b = []
            # print(ex)
            for sentence in ex:
                b.append(self.tokenize(sentence))
            a.append(b)
        return a


        # this parse assumes that we want to keep only one question at the end

    def parse(self, file_data, add_punctuation):
        data, story = [], []

        story2 = []
        i = 0
        with open(file_data, 'r') as f:
            for line in f:
                # print(line)
                tid, text = line.rstrip('\n').split(' ', 1)
                if tid == '1':
                    story = []
                    story2 = []
                    answers = []
                # sentence
                # don't delete period
                if text.endswith('.'):
                    # b= []
                    # for a in text[:-1].split(' '):
                    #    b.append(a[0])
                    # story.extend(b)
                    for a in text[:-1].split():
                        # print(a)
                        # print(type(a))
                        assert not isinstance(a, list)
                        story.append(a)
                        # print(type(story[-1]))
                        # input()

                    # story.extend(text[:-1].strip().split(' '))
                    if add_punctuation:
                        story.append('.')
                # question
                else:
                    # remove any leading or trailing whitespace after splitting
                    query, answer, supporting = (x.strip() for x in text.split('\t'))

                    for a in query[:-1].split(' '):
                        story2.append(a)
                    # story.extend(text[:-1].strip().split(' '))
                    if add_punctuation:
                        story2.append('?')

                    for a in answer.split(','):
                        answers.append(a)
                        story2.extend(['_'])

                    story_f = list(story)
                    story_f.extend(story2)
                    if story_f:
                        data.append((story_f, answers))

                        # NEXT LINES MIGHT BE WRONG -> CHECK WITH RYAN

                        answers = []
                        story2 = []

        # input()
        return data

    def download_from_url(self, url, path):
        """Download file, with logic (from tensor2tensor) for Google Drive"""
        if 'drive.google.com' not in url:
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            with open(path, "wb") as file:
                file.write(r.content)
            return
        print('downloading from Google Drive; may take a few minutes')
        confirm_token = None
        session = requests.Session()
        response = session.get(url, stream=True)
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                confirm_token = v

        if confirm_token:
            url = url + "&confirm=" + confirm_token
            response = session.get(url, stream=True)

        chunk_size = 16 * 1024
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)

    def load_data(self, path=None, root='.data', tasks=[1], tenK=False, add_punctuation=True, data_type='train',
                  outmod=''):

        # loads all asked for tasks into a single file (combining multiple files) and then parses the combined file

        if tenK:
            self.dirname = os.path.join('tasks_1-20_v1-2', 'en-valid-10k')
        else:
            self.dirname = os.path.join('tasks_1-20_v1-2', 'en-valid')

        # print(self.dirname)

        if path is None:
            path = self.download(root)
        # print(path)
        file_data = os.path.join(path, 'collected_' + data_type + outmod + '.txt')
        with open(file_data, 'w') as tf:
            for task in tasks:
                with open(
                        os.path.join(path,
                                     'qa' + str(task) + '_' + data_type + '.txt')) as f:
                    tf.write(f.read())
        return self.parse(file_data, add_punctuation)






