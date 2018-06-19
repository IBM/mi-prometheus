#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sort-of-CLEVR is a simplified version of CLEVR VQA problem """
__author__ = "Tomasz Kornuta"

import h5py
import numpy as np
from PIL import Image, ImageDraw

import progressbar
import logging
logger = logging.getLogger('Sort-of-CLEVR')


import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add path to main project directory - required for testing of the main function and see whether problem is working at all (!)
import os,  sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..','..')) 

import torch
from problems.problem import DataTuple
from problems.image_text_to_class.image_text_to_class_problem import ImageTextToClassProblem, ImageTextTuple


from torch.utils.data.sampler import SubsetRandomSampler


BG_COLOR = (180, 180, 150)
COLOR = [
    (0, 0, 210),
    (0, 210, 0),
    (210, 0, 0),
    (150, 150, 0),
    (150, 0, 150),
    (0, 150, 150),
    # add more colors here if needed
]

N_GRID = 4
NUM_COLOR = len(COLOR)
# the number of objects presented in each image
NUM_SHAPE = 6
# avoid a color shared by more than one objects
NUM_SHAPE = min(NUM_SHAPE, NUM_COLOR)
NUM_Q = 5





class SceneRepresentation:

    def __init__(self, x, y, color, shape):
        self.x = x
        self.y = y
        self.color = color
        self.shape = shape

    def print_graph(self):
        for i in range(len(self.x)):
            s = 'circle' if self.shape[i] else 'rectangle'
            print('{} {} at ({}, {})'.format(color2str(self.color[i]),
                                             s, self.x[i], self.y[i]))



class SortOfCLEVR(ImageTextToClassProblem):
    """
    Sort-of-CLEVR is a simple VQA problem, where the goal is to answer the question regarding a given image.
    Implementation of the generation is based on:
    git@github.com:gitlimlab/Relation-Network-Tensorflow.git
    Improvements:
    - generates scenes with dynamic varying number of objects (2-6)
    - more types of intra- and inter-relational questions
    - more natural interpretation of questions
    - Aux tuple contains scene graph
    """

    def __init__(self, params):
        """
        Initializes Sort-of-CLEVR problem, calls base class initialization, sets properties using the provided parameters.

        :param params: Dictionary of parameters (read from configuration file).
        """

        # Call base class constructors.
        super(SortOfCLEVR, self).__init__(params)

        # Retrieve parameters from the dictionary.
        self.batch_size = params['batch_size']
        self.data_folder = params['data_folder']

        self.img_size = params["img_size"]
        self.dataset_size = params["dataset_size"]


        # Make path absolute.
        if (self.data_folder[0] == '~'):
            # Path to home dir.
            self.data_folder = os.path.expanduser('~') + self.data_folder[1:]
        elif not os.path.isabs(self.data_folder):
            self.data_folder = os.path.abspath(self.data_folder)
        # Create folder - if required.
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)
        print(self.data_folder)

        # Generate the dataset, if not exists.
        # If it exists, simply load it.
        self.generator()

    def generate_batch(self):

        # train_loader a generator: (data, label)
        #(data, label) = next(train_loader)

        # Return DataTuple(!) and an empty (aux) tuple.
        return DataTuple(0, 0), ()


    def color2str(self, color_code):
        return {
            0: 'blue',
            1: 'green',
            2: 'red',
            3: 'yellow',
            4: 'magenta',
            5: 'cyan',
        }[color_code]


    def question_type_template(self, question_code):
        """ Helper function that the string templates a question type. """
        return {
            0: 'What is the shape of the {} object?',
            1: 'Is the {} object closer to the bottom of the image?',
            2: 'Is the {} object closer to the left side of the image?',
            3: 'What is the shape of the object that is nearest to the {} object?',
            4: 'What is the shape of the object that is farthest from the {} object?',
            #5: 'What is the color of the object that is nearest to the {} object?',
            #6: 'What is the color of the object that is farthest from the {} object?',
            #7: 'How many objects have the same shape as the {} object?,
        }[question_code]

    def question2str(self, encoded_question):
        """ Decodes question, i.e. produces a human-understandable string. 
        
        :param color_query: Concatenation of two one-hot vectors: 
          - first one denoting the object of interest (its color), 
          - the second one the question type.
        :return: Question in the form of a string.
        """
        # "Decode" the color_query vector.
        color = np.argmax(encoded_question[:NUM_COLOR])
        question_code = np.argmax(encoded_question[NUM_COLOR:])
        # Return the question as a string.
        return (self.question_type_template(question_code)).format(self.color2str(color))

    #def encode_question(self, color_code, question_code):
        



    def answer2str(self, encoded_answer):
        """ Encodes answer into a string.

        :param encoded_answer: One-hot vector.
        """
        return {
            0: 'blue',
            1: 'green',
            2: 'red',
            3: 'yellow',
            4: 'magenta',
            5: 'cyan',
            6: 'circle',
            7: 'rectangle',
            8: 'yes',
            9: 'no',
        }[np.argmax(encoded_answer)]
  
  #return 'Answer: {}'.format(a_type(av))

    def visualize_iqa(self, img, q, a):
        fig = plt.figure(1)
        plt.title(self.question2str(q))
        plt.xlabel(self.answer2str(a))
        plt.imshow(img)
        plt.show()

    def generate_scene(self):

        img_size = self.img_size

        block_size = int(img_size*0.9/N_GRID)
        shape_size = int((img_size*0.9/N_GRID)*0.7/2)


        # Generate I: [img_size, img_size, 3]
        img = Image.new('RGB', (img_size, img_size), color=BG_COLOR)
        drawer = ImageDraw.Draw(img)
        idx_coor = np.arange(N_GRID*N_GRID)
        np.random.shuffle(idx_coor)
        idx_color_shape = np.arange(NUM_COLOR)
        np.random.shuffle(idx_color_shape)
        coin = np.random.rand(NUM_SHAPE)
        X = []
        Y = []
        for i in range(NUM_SHAPE):
            x = idx_coor[i] % N_GRID
            y = (N_GRID - np.floor(idx_coor[i] / N_GRID) - 1).astype(np.uint8)
            # sqaure terms are added to remove ambiguity of distance
            position = ((x+0.5)*block_size-shape_size+x**2, (y+0.5)*block_size-shape_size+y**2,
                        (x+0.5)*block_size+shape_size+x**2, (y+0.5)*block_size+shape_size+y**2)
            X.append((x+0.5)*block_size+x**2)
            Y.append((y+0.5)*block_size+y**2)
            if coin[i] < 0.5:
                drawer.ellipse(position, fill=COLOR[idx_color_shape[i]])
            else:
                drawer.rectangle(position, fill=COLOR[idx_color_shape[i]])

        # Generate its representation
        color = idx_color_shape[:NUM_SHAPE]
        shape = coin < 0.5
        rep = SceneRepresentation(np.stack(X).astype(np.int),
                            np.stack(Y).astype(np.int), color, shape)
        return np.array(img), rep
        
    def generate_question_matrix(self, rep):
        # Generate questions: [# of shape * # of Q, # of color + # of Q]
        Q = np.zeros((NUM_SHAPE*NUM_Q, NUM_COLOR+NUM_Q), dtype=np.bool)
        for i in range(NUM_SHAPE):
            v = np.zeros(NUM_COLOR)
            v[rep.color[i]] = True
            Q[i*NUM_Q:(i+1)*NUM_Q, :NUM_COLOR] = np.tile(v, (NUM_Q, 1))
            Q[i*NUM_Q:(i+1)*NUM_Q, NUM_COLOR:] = np.diag(np.ones(NUM_Q))
        print("Q = \n",Q)
        return Q

    def generate_answer_matrix(self, rep):
        # Generate answers: [# of shape * # of Q, # of color + 4]
        # # of color + 4: [color 1, color 2, ... , circle, rectangle, yes, no]
        A = np.zeros((NUM_SHAPE*NUM_Q, NUM_COLOR+4), dtype=np.bool)
        for i in range(NUM_SHAPE):
            # Q1: circle or rectangle?
            if rep.shape[i]:
                A[i*NUM_Q, NUM_COLOR] = True
            else:
                A[i*NUM_Q, NUM_COLOR+1] = True

            # Q2: bottom?
            if rep.y[i] > int(self.img_size/2):
                A[i*NUM_Q+1, NUM_COLOR+2] = True
            else:
                A[i*NUM_Q+1, NUM_COLOR+3] = True

            # Q3: left?
            if rep.x[i] < int(self.img_size/2):
                A[i*NUM_Q+2, NUM_COLOR+2] = True
            else:
                A[i*NUM_Q+2, NUM_COLOR+3] = True

            distance = 1.1*(rep.y - rep.y[i]) ** 2 + (rep.x - rep.x[i]) ** 2
            idx = distance.argsort()
            # Q4: the color of the nearest object
            min_idx = idx[1]
            A[i*NUM_Q+3, rep.color[min_idx]] = True
            # Q5: the color of the farthest object
            max_idx = idx[-1]
            A[i*NUM_Q+4, rep.color[max_idx]] = True
        return A

    def generator(self):
        img_size = self.img_size
        dir_name = self.data_folder

        # output files
        f = h5py.File(os.path.join(dir_name, 'data.hy'), 'w')
        #id_file = open(os.path.join(dir_name, 'id.txt'), 'w')

        # progress bar
        bar = progressbar.ProgressBar(maxval=100,
                                    widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                            progressbar.Percentage()])
        bar.start()

        count = 0

        while(count < self.dataset_size):
            # Generate the scene.
            I, scene = self.generate_scene()
            # Generate the image corresponding to the scene.
            A = self.generate_answer_matrix(scene)
            Q = self.generate_question_matrix(scene)
            for j in range(NUM_SHAPE*NUM_Q):
                id = '{}'.format(count)
                #id_file.write(id+'\n')
                grp = f.create_group(id)
                grp['image'] = I
                grp['question'] = Q[j, :]
                grp['answer'] = A[j, :]

                # Show sample.
                #print("Image: ",I)
                #print("Quenstion: ",Q[j, :])
                #print("Answer: ",A[j, :])
                #self.visualize_iqa(I, Q[j, :], A[j, :])


                print("f = {}\n".format(f))
                count += 1
                if count % (self.dataset_size / 100) == 0:
                    bar.update(count / (self.dataset_size / 100))

        # Ok, got the dataset
        bar.finish()
        f.close()
        #id_file.close()
        logger.info('Dataset generated under {} with {} samples.'.format(self.data_folder, self.dataset_size))
        
        return f



if __name__ == "__main__":
    """ Tests sort of CLEVR - generates and displays a sample"""

    # "Loaded parameters".
    params = {'batch_size': 1, 'data_folder': '~/data/sort-of-clevr/', 'dataset_size': 1, 'img_size': 128}

    # Configure logger.
    logging.basicConfig(level=logging.DEBUG)
    logger.debug("params: {}".format(params)) 

    # Create problem object.
    problem = SortOfCLEVR(params)
    # Get generator
    #generator = problem.return_generator()
    # Get batch.
    #(i, q, a) = next(generator)
    # Display single sample (0) from batch.
    #problem.show_sample(i, q, a)
