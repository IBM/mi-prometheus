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
__author__= "Vincent Albouy, Ryan.L McAvoy"

import torch
from miprometheus.utils.problems_utils.language import Language
import torch.utils.data
from tqdm import tqdm
import os
from miprometheus.problems.seq_to_seq.seq_to_seq_problem import SeqToSeqProblem
from miprometheus.utils.app_state import AppState
from miprometheus.problems.seq_to_seq.seq_to_seq_problem import SeqToSeqProblem
from miprometheus.utils.loss.masked_cross_entropy_loss import MaskedCrossEntropyLoss
import torch.nn as nn


class BABI(SeqToSeqProblem):
    """
    Problem Class for loading bAbi QA data set using Torchtext
    
    Inherits from SeqToSeqProblem
    
    """

    def __init__(self, params):
        """
   
        Initializes BABI QA problem, calls base class initialization, sets properties using the provided parameters.

        :param params: Dictionary of parameters (read from configuration file).
        
        """
        super(BABI).__init__()

        self.directory = './'

        # boolean: is it training phase?
        self.data_type = params['data_type']

        self.use_batches = params['batch_size']

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

        self.use_mask = False

        self.urls = ['http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz']

        self.name = 'BABIDataset'

        self.dirname = ''

        self.data = self.load_data( tasks=self.tasks, tenK=self.tenK, add_punctuation=True , data_type = self.data_type)

        #create an object language from Language class - This object will be used to create the words embeddings
        self.language = Language('lang')

        self.default_values = {'input_item_size': 38, 'output_item_size': 231}

        self.data_definitions = {'sequences': {'size': [-1, -1, self.memory_size], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1], 'type': [torch.Tensor]},
                                 'current_question': {'size': [-1, 1], 'type': [list, str]},
                                 'masks': {'size': [-1], 'type': [torch.Tensor]},
                                 }


        #building the embeddings
        if self.one_hot_embedding:
            self.dictionaries, self.itos_dict = self.build_dictionaries_one_hot()
        else:
            self.dictionaries, self.itos_dict = self.build_dictionaries()

        if self.use_mask:
            self.loss_function = MaskedBCEWithLogitsLoss()
        else:
            self.loss_function = nn.NLLLoss()


    def __len__(self):
        """Return the length of the questions set"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Getter method to access the dataset and return a sample.
        :param idx: index of the sample to return.
        :return: sample = {'sequence': story, 'targets': target, 'current_question': current_question, "mask": mask}
        """
        #get current question with indices idx
        current_question = self.data[idx]

        written_story, written_answers = current_question

        current_question = [" ".join(written_story), " ".join(written_answers)]

        story = self.embed_batch(written_story)

        answer = self.to_dictionary_indexes(self.dictionaries, written_answers)

        mask = torch.zeros((story.shape[0])).type(AppState().ByteTensor)
        k =0

        target = torch.zeros((story.shape[0])).type(AppState().LongTensor) 

        for i, word in enumerate(current_question[0].split(' ')):
            if word == '_':
                mask[i] = 1
                target[i] = answer[k]
                k=k+1

        #make a dictionnary with all the outputs
        data_dict = self.create_data_dict()
        data_dict['sequences'] = story
        data_dict['targets'] = target
        data_dict['current_question'] = current_question
        data_dict['masks'] = mask

        #return the fina DataDict
        return data_dict

    def collate_babi(self, batch):

        """            
               Collate method that create batch from samples.
               :param batch.
               :return: return {'sequence': sequence, 'targets': targets, 'current_question': current_question, "mask": mask} 
               """
        # get sizes
        context_length = max(d["sequences"].shape[0] for d in batch)
        answer_length = max(d["targets"].shape[0] for d in batch)
        batch_size = len(batch)
        word_size = batch[0]["sequences"].shape[-1]

        # create placeholders
        sequences = torch.zeros((batch_size, context_length, word_size)).type(AppState().dtype)
        targets = torch.zeros((batch_size, answer_length)).type(AppState().LongTensor)
        mask = torch.zeros((batch_size, answer_length)).type(AppState().ByteTensor)

        # padded data
        current_question = []
        for i, d in enumerate(batch):
            c_shape = d["sequences"].shape
            a_shape = d["targets"].shape
            sequences[i, :c_shape[0], :c_shape[1]] = d["sequences"]
            targets[i, :a_shape[0]] = d["targets"]
            mask[i, :a_shape[0]] = d["masks"]
            current_question.append(d["current_question"])

            # make a dictionnary with all the outputs
            data_dict = self.create_data_dict()
            data_dict['sequences'] = sequences
            data_dict['targets'] = targets
            data_dict['current_question'] = current_question
            data_dict['masks'] = mask

        # return the fina DataDict
        return data_dict


    def build_dictionaries_one_hot(self):

        """Creates the word embeddings for BABI QA with one hot vectors

                - 1. Collects all datasets word
                - 2. Uses Language object to create the embeddings

                 If it is the first time you run this code, it will take longer to load the embedding from torchtext
                 """
        #load data
        data = self.load_data(tasks=self.tasks, tenK=self.tenK, add_punctuation=True, data_type='train', outmod="one_hot")
        data = data + self.load_data(tasks=self.tasks, tenK=self.tenK, add_punctuation=True, data_type='valid',
                                     outmod="one_hot")
        data = data + self.load_data(tasks=self.tasks, tenK=self.tenK, add_punctuation=True, data_type='test',
                                     outmod="one_hot")

        # make placeholders dictionnaries with special caracters
        answ_to_ix = {".": 0, "?": 1, "_": 2}
        itos_d = [".", "?", "_"]

        # display a progress bar while going through the data
        self.fix_length = 0
        for q in tqdm(data):
            story, answers = q
            self.fix_length = max(self.fix_length, len(story))

            # go through all the stories
            for answer in story:
                a = answer.lower()
                if a not in answ_to_ix:
                    ix = len(answ_to_ix)
                    answ_to_ix[a] = ix
                    itos_d.append(a)
                    # print(a,ix)

            #go through all the answers
            for answer in answers:
                a = answer.lower()
                if a not in answ_to_ix:
                    ix = len(answ_to_ix)
                    answ_to_ix[a] = ix
                    itos_d.append(a)
                    # print(a, ix)

        #return the corresponding dictionnaries
        ret = (answ_to_ix)
        return ret, itos_d

    def build_dictionaries(self):

        """Creates the word embeddings BABI QA 

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

        # make placeholders dictionnaries with special caracters
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

            # go through all the stories
            for answer in story:
                a = answer.lower()
                if a not in answ_to_ix:
                    ix = len(answ_to_ix)
                    answ_to_ix[a] = ix
                    itos_d.append(a)
                    # print(a,ix)

            # go through all the answers
            for answer in answers:
                a = answer.lower()
                if a not in answ_to_ix:
                    ix = len(answ_to_ix)
                    answ_to_ix[a] = ix
                    itos_d.append(a)
                    # print(a, ix)

        """ build embeddings from the chosen database / Example: glove.6B.100d """

        self.language.build_pretrained_vocab(text, vectors=self.embedding_type, tokenize=self.tokenize)

        # return the corresponding dictionnaries
        ret = (answ_to_ix)
        return ret, itos_d



    def download_from_url(self, url, path):

        """Download file, with logic (from tensor2tensor) for Google Drive"""

        #get url and write file to path
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

        #open file + write chunks
        chunk_size = 16 * 1024
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)

    def load_data(self, path=None, root='.data', tasks=[1], tenK=False, add_punctuation=True, data_type='train',
                  outmod=''):

        """loads all asked for tasks into a single file (combining multiple files) and then parses the combined file"""

        if tenK:
            self.dirname = os.path.join('tasks_1-20_v1-2', 'en-valid-10k')
        else:
            self.dirname = os.path.join('tasks_1-20_v1-2', 'en-valid')

        if path is None:
            path = self.download(root)

        file_data = os.path.join(path, 'collected_' + data_type + outmod + '.txt')
        with open(file_data, 'w') as tf:
            for task in tasks:
                with open(
                        os.path.join(path,
                                     'qa' + str(task) + '_' + data_type + '.txt')) as f:
                    tf.write(f.read())
        return self.parse(file_data, add_punctuation)

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
        #get path
        path = os.path.join(root, self.name)
        check = path if check is None else check

        #download data
        if not os.path.isdir(check):
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

                #unzip the data
                if ext == '.zip':
                    with zipfile.ZipFile(zpath, 'r') as zfile:
                        print('extracting')
                        zfile.extractall(path)

                # tarfile cannot handle bare .gz files
                elif ext == '.tgz' or ext == '.gz' and ext_inner == '.tar':
                    with tarfile.open(zpath, 'r:gz') as tar:
                        dirs = [member for member in tar.getmembers()]
                        tar.extractall(path=path, members=dirs)

                #in case it is a gz file
                elif ext == '.gz':
                    with gzip.open(zpath, 'rb') as gz:
                        with open(zroot, 'wb') as uncompressed:
                            shutil.copyfileobj(gz, uncompressed)

        #Return path to extracted dataset
        return os.path.join(path, self.dirname)


    def parse(self, file_data, add_punctuation):

        """This method is parsing the file
               :param file_data : data file to  be parsed
               :param add_punctuation : boolean to decide wether we add punctuation 
               :return: data : Parsed data
           
        """
        #make empty lists
        data, story,  story2 = [],[],[]
        i = 0

        #open file
        with open(file_data, 'r') as f:
            for line in f:
                # print(line)
                tid, text = line.rstrip('\n').split(' ', 1)
                if tid == '1':
                    story = []
                    story2 = []
                    answers = []
                # don't delete period
                if text.endswith('.'):
                    for a in text[:-1].split():
                        assert not isinstance(a, list)
                        story.append(a)
                    if add_punctuation:
                        story.append('.')
                else:
                    # remove any leading or trailing whitespace after splitting
                    query, answer, supporting = (x.strip() for x in text.split('\t'))

                    for a in query[:-1].split(' '):
                        story2.append(a)
                    if add_punctuation:
                        story2.append('?')
                    for a in answer.split(','):
                        answers.append(a)
                        story2.extend(['_'])

                    story_f = list(story)
                    story_f.extend(story2)
                    if story_f:
                        data.append((story_f, answers))

                        #Set answers and story back to empty lists
                        answers = []
                        story2 = []

        return data



if __name__ == "__main__":

    """Unitest that generates a batch and displays a sample """

    babi_tasks = list(range(1, 21))

    params = {'directory': '/', 'tasks': babi_tasks,'data_type': 'train', 'batch_size': 10,'embedding_type' :'glove.6B.100d', 'ten_thousand_examples': True, 'one_hot_embedding': True, 'truncation_length':50 }



    babi = BABI(params)
    sample=babi[10]
    print(sample['sequences'].size())
    print(sample['current_question'])
    print(len(sample['current_question'][0].split(' ')))
    print(babi[0])
    print('__getitem__ works.')


    # wrap DataLoader on top of this Dataset subclass
    from torch.utils.data.dataloader import DataLoader

    batch_size = 1
    dataloader = DataLoader(dataset=babi, collate_fn=babi.collate_babi,
                            batch_size=batch_size, shuffle=True, num_workers=0)

    # try to see if there is a speed up when generating batches w/ multiple workers
    import time
    s = time.time()
    #for i, batch in enumerate(dataloader):
    #     print('Batch # {} - {}'.format(i, type(batch)))
    # print('Number of workers: {}'.format(dataloader.num_workers))
    #print('time taken to exhaust the dataset for a batch size of {}: {}s'.format(batch_size, time.time() - s))

    batch = next(iter(dataloader))
    print(batch)
    print('Unit test completed')
    exit()
