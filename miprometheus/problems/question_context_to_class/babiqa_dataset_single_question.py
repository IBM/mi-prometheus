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
__author__= "Vincent Albouy"

import torch
# Add path to main project directory - required for testing of the main function and see whether problem is working at all (!)
import os,  sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..','..'))
from utils.problems_utils.language import Language
from problems.question_context_to_class.question_context_to_class_problem import QuestionContextToClass
from torch.utils.data import Dataset
import torch.utils.data
from problems.seq_to_seq.text2text.text_to_text_problem import TextToTextProblem
from utils.app_state import AppState


class BABI(QuestionContextToClass, Dataset):
    """
    Problem Class for loading bAbi QA data set using Torchtext
    
    Inherits from text_to_text_problem.py and utils.data
    
    """

    def __init__(self, params):
        """
   
        Initializes BABIQA problem, calls base class initialization, sets properties using the provided parameters.

        :param params: Dictionary of parameters (read from configuration file).
        
        """
        super(BABI).__init__()

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

        self.name = 'BABIDataset'

        self.dirname = ''

        self.data = self.load_data( tasks=self.tasks, tenK=self.tenK, add_punctuation=True , data_type = self.data_type)

        #create an object language from Language class - This object will be used to create the words embeddings
        self.language = Language('lang')

        #building the embeddings
        if self.one_hot_embedding:
            self.dictionaries, self.itos_dict = self.build_dictionaries_one_hot()
        else:
            self.dictionaries, self.itos_dict = self.build_dictionaries()


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
        #print(string_story)
        for i, word in enumerate(current_question[0].split(' ')):
            if word == '_':
                mask[i] = 1
                target[i] = answer[k]
                #print(word, i)
                k=k+1

        #make a dictionnary with all the outputs
        sample = {'sequence': story, 'targets': target, 'current_question': current_question, "mask": mask}

        return sample


    def collate_babi(self, batch):

        """            
               Collate method that create batch from samples.
               :param batch.
               :return: return {'sequence': sequence, 'targets': targets, 'current_question': current_question, "mask": mask}
               
               """


        context_length = max(d["sequence"].shape[0] for d in batch)
        answer_length = max(d["targets"].shape[0] for d in batch)
        batch_size = len(batch)
        word_size = batch[0]["sequence"].shape[-1]

        sequence = torch.zeros((batch_size, context_length, word_size)).type(AppState().dtype)
        targets = torch.zeros((batch_size, answer_length)).type(AppState().LongTensor)
        mask = torch.zeros((batch_size, answer_length)).type(AppState().ByteTensor)

        current_question = []
        for i, d in enumerate(batch):
            c_shape = d["sequence"].shape
            a_shape = d["targets"].shape
            sequence[i, :c_shape[0], :c_shape[1]] = d["sequence"]
            targets[i, :a_shape[0]] = d["targets"]
            mask[i, :a_shape[0]] = d["mask"]
            current_question.append(d["current_question"])

        return {'sequence': sequence, 'targets': targets, 'current_question': current_question, "mask": mask}



if __name__ == "__main__":

    """Unitest that generates a batch and displays a sample """

    babi_tasks = list(range(1, 21))

    params = {'tasks': babi_tasks,'data_type': 'train', 'batch_size': 10,'embedding_type' :'glove.6B.100d', 'ten_thousand_examples': True, 'one_hot_embedding': True, 'truncation_length':50 }

    batch_size=10

    babi = BABI(params)
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
