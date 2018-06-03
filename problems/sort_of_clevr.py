#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sort-of-Clevr is a simplified version of Clevr """
__author__ = "Mikyas Desta"

import matplotlib
matplotlib.use('agg')
import pickle
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from subprocess import call
from algorithmic_sequential_problem import AlgorithmicSequentialProblem
from algorithmic_sequential_problem import DataTuple


@AlgorithmicSequentialProblem.register
class Sort_of_clevr(AlgorithmicSequentialProblem):
    """

    """

    def __init__(self, params):
        """
        Constructor - stores parameters.

        :param params: Dictionary of parameters.
        """
        # Retrieve parameters from the dictionary.
        self.batch_size = params['batch_size']
        self.batch_idx = params['batch_idx']
        self.train_or_test = params['train_or_test']
        self.rel_or_norel= params['rel_or_norel']
        self.read_pickle_file()
        rel_train, rel_test, norel_train, norel_test = self.load_data()
        self.rel_train = rel_train
        self.rel_test = rel_test
        self.norel_train = norel_train
        self.norel_test = norel_test

        self.dtype = torch.FloatTensor


    def read_pickle_file(self):
        objects = []
        try:
            with (open("sort_of_clevr/sort-of-clevr.pickle", "rb")) as openfile:
                while True:
                    try:
                        objects.append(pickle.load(openfile))
                    except EOFError:
                        break
        except IOError:
            call(["python3", "sort_of_clevr/sort_of_clevr_generator.py"])


    """This part of the code(load data) was taken from https://github.com/kimhc6028/relational-networks you can simply rewrite it.
    I didn't have time to rewrite it. """
    def load_data(self):
        print('loading data...')
        dirs = './sort_of_clevr'
        filename = os.path.join(dirs, 'sort-of-clevr.pickle')
        with open(filename, 'rb') as f:
            train_datasets, test_datasets = pickle.load(f)
        rel_train = []
        rel_test = []
        norel_train = []
        norel_test = []
        print('processing data...')

        for img, relations, norelations in train_datasets:
            img = np.swapaxes(img, 0, 2)
            for qst, ans in zip(relations[0], relations[1]):
                rel_train.append((img, qst, ans))
            for qst, ans in zip(norelations[0], norelations[1]):
                norel_train.append((img, qst, ans))

        for img, relations, norelations in test_datasets:
            img = np.swapaxes(img, 0, 2)
            for qst, ans in zip(relations[0], relations[1]):
                rel_test.append((img, qst, ans))
            for qst, ans in zip(norelations[0], norelations[1]):
                norel_test.append((img, qst, ans))

        return (rel_train, rel_test, norel_train, norel_test)

    def show_sample(self, image, question,answer ):
        plt.imshow(image)
        plt.show()
        matplotlib.pyplot.show()
        print("answer", answer)
        print("shape of question", np.shape(question))
        print("shape of answer",np.shape(answer))
        print("image shape", np.shape(image))


    def generate_batch(self):
        """

        """
        if self.train_or_test == True and self.rel_or_norel == True:
            data = self.rel_train
        elif self.train_or_test == True and self.rel_or_norel == False:
            data = self.norel_train
        elif self.train_or_test == False and self.rel_or_norel == True:
            data = self.rel_test
        else:
            data = self.norel_test

        img = np.asarray(data[0][self.batch_size * self.batch_idx:self.batch_size * (self.batch_idx + 1)])
        qst = np.asarray(data[1][self.batch_size * self.batch_idx:self.batch_size * (self.batch_idx + 1)])
        ans = np.asarray(data[2][self.batch_size * self.batch_idx:self.batch_size * (self.batch_idx + 1)])
        img_pt = torch.from_numpy(img).type(self.dtype)
        qst_pt = torch.from_numpy(qst).type(self.dtype)
        ans_pt = torch.from_numpy(ans).type(self.dtype)


        # Return data tuple.
        return DataTuple(img_pt, qst_pt, ans_pt)


if __name__ == "__main__":
    """ Tests sort of clevr - generates and displays a sample"""

    # "Loaded parameters".
    params = {'batch_size': 1,'batch_idx': 1, 'train_or_test': True, 'rel_or_norel': True, 'bias': 0.5}
    # Create problem object.
    problem = Sort_of_clevr(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    (i, q, a) = next(generator)
    # Display single sample (0) from batch.
    problem.show_sample(i, q, a)
