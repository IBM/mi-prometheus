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

"""bAbiQA.py: contains code for loading the babi dataset (based on the parsing used in torchtext)"""
__author__= "Ryan L. McAvoy, Vincent Albouy"

import torch
from torchtext import datasets,data
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

# Add path to main project directory - required for testing of the main function and see whether problem is working at all (!)
import os,  sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..','..')) 

from utils.problems_utils.language import Language
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

class BABIDatasets(Dataset):
    """

    class Dataset(object): An abstract class representing a Dataset.
    All other datasets should subclass it. All subclasses should override __len__, that provides the size of the dataset
     , and __getitem__, supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self,params):
        """
        Instantiate a ClevrDataset object.
        :param train: Boolean to indicate whether or not the dataset is constructed for training.
        :param clevr_dir:  clevr directory
        :param clevrhumans_dir: clevr humans directory
        :param clevrhumans: Boolean - is it training phase?
        
        """
        super(BABIDataset).__init__()

        # clevr directory
        self.directory = './'

        # boolean: is it training phase?
        self.data_type = params['data_type']

        self.use_batches = batch_size > 1

        # task number to train on
        self.tasks = params['tasks']

        self.tenK = params['ten_thousand_examples']

        self.one_hot_embedding = params['one_hot_embedding']

        self.batch_size = params['batch_size']

        self.memory_size = params['truncation_length']

        self.embedding_type = params['embedding_type']

        self.init_token = '<sos>'

        self.pad_token = '<pad>'

        self.eos_token = '<eos>'

        self.urls = ['http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz']

        self.name = ''

        self.dirname = ''

        self.data = self.load_data( tasks=self.tasks, tenK=self.tenK, add_punctuation=True , data_type = self.data_type)

        #create an object language from Language class - This object will be used to create the words embeddings
        self.language = Language('lang')

        #building the embeddings
        if self.one_hot_embedding:
            self.dictionaries, self.itos_dict = self.build_dictionaries_one_hot()
        else:
            self.dictionaries, self.itos_dict = self.build_dictionaries()

    def build_dictionaries_one_hot(self):
              
        tasks = list(range(1,21))
 
        data = self.load_data( tasks=tasks, tenK=self.tenK, add_punctuation=True , data_type = 'train', outmod = "one_hot")
        data = data + self.load_data( tasks=tasks, tenK=self.tenK, add_punctuation=True , data_type = 'valid', outmod = "one_hot")
        data = data + self.load_data( tasks=tasks, tenK=self.tenK, add_punctuation=True , data_type = 'test', outmod = "one_hot")

        answ_to_ix = {".": 0, "?": 1, "_": 2 } 
        itos_d =[".", "?", "_"]
        for q in tqdm(data):

            story, answers = q

            for answer in story:
                a = answer.lower()
                if a not in answ_to_ix:
                    ix = len(answ_to_ix)
                    answ_to_ix[a] = ix
                    itos_d.append(a)
                    print(a,ix)

          
            for answer in answers:
                a = answer.lower()
                if a not in answ_to_ix:
                    ix = len(answ_to_ix)
                    answ_to_ix[a] = ix
                    itos_d.append(a)
                    print(a, ix)

        ret = ( answ_to_ix)
        return ret, itos_d

    def build_dictionaries(self):

        """Creates the word embeddings    
        - 1. Collects all datasets word
        - 2. Uses Language object to create the embeddings    
         If it is the first time you run this code, it will take longer to load the embedding from torchtext
         """

        print(' ---> Constructing the dictionaries with word embedding, may take some time ')

        #making an empty list of words meant to store all possible datasets words
        text = []
        tasks = list(range(1,21))

        data = self.load_data( tasks=tasks, tenK=self.tenK, add_punctuation=True , data_type = 'train', outmod = "embedding")
        data = data + self.load_data( tasks=tasks, tenK=self.tenK, add_punctuation=True , data_type = 'valid', outmod = "embedding")
        data = data + self.load_data( tasks=tasks, tenK=self.tenK, add_punctuation=True , data_type = 'test', outmod = "embedding")


        answ_to_ix = {".": 0, "?": 1, "_": 2 }
        itos_d =[".", "?", "_"]
        # load all words from training data to a list named words_list[]
        self.fix_length = 0
        for q in tqdm(data):
            # display a progress bar
            story, answers = q
            self.fix_length =max(self.fix_length, len(story))

            for word in story:
                text.extend([word.lower()])
           
            for answer in story:
                a = answer.lower()
                if a not in answ_to_ix:
                    ix = len(answ_to_ix)
                    answ_to_ix[a] = ix
                    itos_d.append(a)
                    print(a,ix)

            for answer in answers:
                a = answer.lower()
                if a not in answ_to_ix:
                    ix = len(answ_to_ix)
                    answ_to_ix[a] = ix
                    itos_d.append(a)
                    print(a, ix)


        """ build embeddings from the chosen database / Example: glove.6B.100d """

        self.language.build_pretrained_vocab(text, vectors=self.embedding_type, tokenize = self.tokenize)

        ret = ( answ_to_ix)
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
        for ind in int_sentence[0,:]:
            sentences.append(self.itos_dict[ind])
        return sentences

    def tokenize(self, sentence):
         return sentence.split(' ')

    def __len__(self):
        """Return the length of the questions set"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Getter method to access the dataset and return a sample.
        :param idx: index of the sample to return.
        :return: {'image': image, 'question': question, 'answer': answer, 'current_question': current_question}
        """
        #get current question with indices idx
        current_question = self.data[idx]
        story, answers = current_question
        #print(story)
        current_question = (" ".join(story), " ".join(answers))

        #build story
        story = self.embed_batch(story)
        #question = self.language.embed_sentence(question)

        #do I need to embed answers in this way
        print(answers)
        answer = self.to_dictionary_indexes(self.dictionaries, answers)

        #make a dictionnary with all the outputs
        sample = {'context': story, 'targets': answer, 'current_question': current_question}

        return sample

    def embed_sentence_one_hot(self, sentence):
        """
        Embed an entire sentence using a pretrained embedding
        :param sentence: A string containing the words to embed
        :returns: FloatTensor of embedded vectors [max_sentence_length, embedding size]
        """
        size_hot = len(self.dictionaries)
        outsentence = torch.zeros((len(sentence.split(" ")), size_hot))

        #embed word by word
        for i, word in enumerate(sentence.split(" ")):
            index = self.dictionaries[word.lower()]
            outsentence[i,index] = 1

        return outsentence

    def embed_batch(self, minibatch):

        #embed a whole batch
        ex= minibatch
        sentence = " ".join(ex)
        if self.one_hot_embedding:
            sent_embed = self.embed_sentence_one_hot(sentence)
        else:
            sent_embed = self.language.embed_sentence(sentence)

        return sent_embed


    def detokenize_story(self, minibatch):
        a = []
        for ex in minibatch:
           b= []
           #print(ex)
           for sentence in ex:
               b.append(" ".join(sentence))
           a.append(b)
        return a


    def tokenize_story(self, minibatch):
        a = []
        for ex in minibatch:
           b= []
           #print(ex)
           for sentence in ex:
               b.append(self.tokenize(sentence))
           a.append(b)
        return a

    #this parse assumes that we want to split up substories and have only one question at the end
    def parseOld(self, file_data, only_supporting):
        data, story = [], []
        with open(file_data, 'r') as f:
            for line in f:
                #print(line)
                tid, text = line.rstrip('\n').split(' ', 1)
                if tid == '1':
                    story = []
                # sentence
                if text.endswith('.'):
                    story.append(text[:-1])
                # question
                else:
                    # remove any leading or trailing whitespace after splitting
                    query, answer, supporting = (x.strip() for x in text.split('\t'))
                    if only_supporting:
                        substory = [story[int(i) - 1] for i in supporting.split()]
                    else:
                        substory = [x for x in story if x]
                    data.append((substory, query[:-1], answer))    # remove '?'
                    story.append("")
        return data

    #this parse assumes that we want to keep only one question at the end
    def parseOld2(self, file_data, add_punctuation):
        data, story, question = [], [], []
        i=0
        with open(file_data, 'r') as f:
            for line in f:
                #print(line)
                tid, text = line.rstrip('\n').split(' ', 1)
                if tid == '1':
                    story = []
                    answers = []
                    question =[]
                # sentence
                # don't delete period
                if text.endswith('.'):
                    #b= []
                    #for a in text[:-1].split(' '):
                    #    b.append(a[0])
                    #story.extend(b)
                    for a in text[:-1].split():
                        #print(a)
                        #print(type(a))
                        assert not isinstance(a,list)
                        story.append(a)
                        #print(type(story[-1])) 
                        #input()

                    #story.extend(text[:-1].strip().split(' '))
                    if add_punctuation:
                        story.append('.')
                # question
                else:
                    # remove any leading or trailing whitespace after splitting
                    query, answer, supporting = (x.strip() for x in text.split('\t'))

                    #print(answer)
                    #input()
                    #answers.append(answer)
                    for a in answer.split(','):
                        answers.append(a)

                    #don't detete question marks
                    #story.extend(query[:-1].split(' '))
                    #if add_punctuation:
                    #    story.extend(['?'])

                    for a in query[:-1].split(' '):
                        question.append(a)
                    #story.extend(text[:-1].strip().split(' '))
                    if add_punctuation:
                        question.append('?')



                    for a in answer.split(','):
                        question.extend(['_'])
                    
                    substory = [x for x in story if x]
                    substory.extend(question)    
                    if story:
                        #substory = [x.split() for x in story if x]
                        data.append((substory, answers))
                        answers = []
                        question =[]
        return data



    #this parse assumes that we want to keep whole stories with multiple questions
    def parse(self, file_data, add_punctuation):
        data, story = [], []
        i=0
        with open(file_data, 'r') as f:
            for line in f:
                #print(line)
                tid, text = line.rstrip('\n').split(' ', 1)
                if tid == '1':
                    if story:
                        #substory = [x.split() for x in story if x]
                        data.append((story, answers))
                    story = []
                    answers = []
                # sentence
                # don't delete period
                if text.endswith('.'):
                    for a in text[:-1].split():
                        #print(a)
                        #print(type(a))
                        assert not isinstance(a,list)
                        story.append(a)

                    #story.extend(text[:-1].strip().split(' '))
                    if add_punctuation:
                        story.append('.')
                # question
                else:
                    # remove any leading or trailing whitespace after splitting
                    query, answer, supporting = (x.strip() for x in text.split('\t'))

                    #story.extend(['?'])
                    for a in query[:-1].split(' '):
                        story.append(a)
                    #story.extend(text[:-1].strip().split(' '))
                    if add_punctuation:
                        story.append('?')

                    for a in answer.split(','):
                        answers.append(a)    
                        story.extend(['_'])
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


    def download(self, root, check=None):
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
    
    def load_data(self, path=None, root='.data', tasks=[1], tenK=False, add_punctuation=True, data_type = 'train', outmod='' ):

  
        if tenK:
            self.dirname = os.path.join('tasks_1-20_v1-2', 'en-valid-10k')
        else:
            self.dirname = os.path.join( 'tasks_1-20_v1-2', 'en-valid')

        #print(self.dirname)

        if path is None:
            path = self.download(root)
        #print(path)
        file_data = os.path.join(path, 'collected_'+ data_type+outmod+'.txt')
        with open(file_data, 'w') as tf:
            for task in tasks:
                with open(
                    os.path.join(path,
                        'qa' + str(task) + '_'+data_type+'.txt')) as f:
                    tf.write(f.read())
        return self.parse(file_data, add_punctuation)

    # a simple custom collate function, just to show the idea
    def my_collate(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        target = torch.LongTensor(target)
        return [data, target]


    def pad_batch(self, minibatch):
        #memory size is the number of padded sentence
        #fix_length is the length of each padded sentence
              
        
        #self.fix_length = max(max(max(len(y) for y in x) for x in ex) for ex in minibatch) 
        if not self.use_batches:
            return minibatch

        padded=[]
        for ex in minibatch:
            ex.extend([self.pad_token]*self.fix_length)
            padded.append(ex)

        return padded



    def pad_stories(self, minibatch):

        if self.use_batches:
            minibatch.extend([self.pad_token]*self.fix_length)

        return minibatch

    def pad_storiesOld(self, minibatch):


        self.fix_length = max(max(max(len(y) for y in x) for x in ex) for ex in minibatch)
        padded = []
        for ex in minibatch:
        # sentences are indexed in reverse order and truncated to memory_size
            nex1 = ex[::-1]
            #print(nex1)
            nex = nex1[:self.memory_size]
            #print(nex)
            padded.append(self.pad_sentences(nex) +
             [[self.pad_token] * self.fix_length] * (self.memory_size - len(nex)))
        return padded


    def pad_sentences(self, minibatch):
        """Pad a batch of examples using this field.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True` and `self.sequential` is `True`, else just
        returns the padded list. If `self.sequential` is `False`, no padding is applied.
        """

        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2

        padded, lengths = [], []
        for x in minibatch:
            padded.append(
                ([] if self.init_token is None else [self.init_token]) +
                list(x[:max_len]) +
                ([] if self.eos_token is None else [self.eos_token]) +
                [self.pad_token] * max(0, max_len - len(x)))
        return padded


if __name__ == "__main__":

    """Unitest that generates a batch and displays a sample """

    babi_tasks = list(range(1, 21))

    params = {'tasks': babi_tasks,'data_type': 'train', 'batch_size': 10,'embedding_type' :'glove.6B.100d', 'ten_thousand_examples': True, 'one_hot_embedding': True, 'truncation_length':50 }

    batch_size=10

    babi = BABIDataset(params)
    sample=babi[10]
    print(sample)
    print(babi[0])
    print('__getitem__ works.')

    # wrap DataLoader on top of this Dataset subclass
    from torch.utils.data.dataloader import DataLoader

    dataloader = DataLoader(dataset=babi, collate_fn=babi.collate_babi,
                            batch_size=batch_size, shuffle=True, num_workers=4)

    # try to see if there is a speed up when generating batches w/ multiple workers
    import time

    s = time.time()
    for i, batch in enumerate(dataloader):
        # print('Batch # {} - {}'.format(i, type(batch)))
        pass

    print('Number of workers: {}'.format(dataloader.num_workers))
    print('time taken to exhaust the dataset for a batch size of {}: {}s'.format(batch_size, time.time() - s))
    batch = next(iter(dataloader))
    print(batch)
    print('Unit test completed')
    exit()


