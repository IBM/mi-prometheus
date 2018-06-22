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

import random
import collections
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add path to main project directory - required for testing of the main function and see whether problem is working at all (!)
import os,  sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..','..')) 

import torch
from problems.problem import DataTuple
from problems.image_text_to_class.image_text_to_class_problem import ImageTextToClassProblem, ImageTextTuple


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

# "Hyperparameters"
N_GRID = 4
NUM_COLORS = len(COLOR)
NUM_QUESTIONS = 7
# Objects are characterised by colors, so cannot have more objects than colors.
MAX_NUM_OBJECTS = min(6, NUM_COLORS)




_SceneDescriptionTuple = collections.namedtuple('_SceneDescriptionTuple', ('scene_descriptions'))

class SceneDescriptionTuple(_SceneDescriptionTuple):
    """Tuple used by storing batches of scene descriptions - as strings. """
    __slots__ = ()
    



class ObjectRepresentation:
    """ Class storing features of the object being present in a given scene. """
    def __init__(self, x, y, color, shape):
        self.x = x
        self.y = y
        self.color = color
        self.shape = shape


class SortOfCLEVR(ImageTextToClassProblem):
    """
    Sort-of-CLEVR is a simple VQA problem, where the goal is to answer the question regarding a given image.
    Implementation of the generation is inspired by:
    git@github.com:gitlimlab/Relation-Network-Tensorflow.git
    Improvements:
    - generates scenes with dynamic varying number of objects (2-6)
    - more types of intra- and inter-relational questions
    - more natural interpretation of questions
    - 
    - Aux tuple containing the scene graphs
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

        self.img_size = params["img_size"]
        self.dataset_size = params["dataset_size"]

        # Shuffle indices.
        self.shuffle = params.get('shuffle', True)

        # Get path
        data_folder = params['data_folder']
        data_filename = params['data_filename']

        # Make path absolute.
        if (data_folder[0] == '~'):
            # Path to home dir.
            data_folder = os.path.expanduser('~') + data_folder[1:]
        elif not os.path.isabs(data_folder):
            # Path to current dir.
            data_folder = os.path.abspath(data_folder)

        # Ok, try to load the file.
        self.pathfilename = os.path.join(data_folder, data_filename)

        try:
            if params.get("regenerate", False):
                raise Exception("Must regenerate... must regenerate...")
            self.data = h5py.File(self.pathfilename, 'r')
        except:
            logger.warning('File {} in {} not found. Generating new Sort-of-CLEVR dataset file'.format(data_filename, data_folder))
            # Create folder - if required.
            if not os.path.exists(data_folder):
                os.mkdir(data_folder)
                
            # Generate the dataset, if not exists.
            # If it exists, simply load it.
            self.generate_h5py_dataset()

            # Load the file.
            self.data = h5py.File(self.pathfilename, 'r')

        logger.info("Loaded {} Sort-of-CLEVR samples from file {}".format(len(self.data), self.pathfilename))
        
        # Generate list of indices (strings).
        self.ids = ['{}'.format(i) for i in range(len(self.data))]


    def generate_batch(self):
        """ Generates batch.
        
        :return: DataTuple and AuxTuple object.
        """
        # Shuffle indices.
        if self.shuffle:
            random.shuffle(self.ids)
        # Get batch of indices.
        batch_ids = self.ids[:self.batch_size]
        #print(batch_ids)

        # Get batch.
        images = []
        questions = []
        answers = []
        scenes = []

        for id in batch_ids:
            group = self.data[id]
            # Process data
            images.append(group['image'].value/255.) 
            questions.append(group['question'].value.astype(np.float32)) 
            answers.append(group['answer'].value.astype(np.float32)) 
            scenes.append(group['scene_description'].value)

        # Generate tuple with inputs
        inputs = ImageTextTuple( np.stack(images, axis=0), np.stack(questions, axis=0))
        targets = np.stack(answers, 0)

        # TODO: aux tuple.
        aux_tuple = SceneDescriptionTuple(scenes)

        # Return DataTuple(!) and an aux tuple with scene descriptions.
        return DataTuple(inputs, targets), aux_tuple


    def color2str(self, color_code):
        " Decodes color and returns it as a string. "
        return {
            0: 'blue',
            1: 'green',
            2: 'red',
            3: 'yellow',
            4: 'magenta',
            5: 'cyan',
        }[color_code]


    def shape2str(self, shape_code):
        " Decodes shape and returns it as a string. "
        return {
            0: 'rectangle',
            1: 'circle',
        }[shape_code]


    def question_type_template(self, question_code):
        """ Helper function that the string templates a question type. """
        return {
            0: 'What is the shape of the {} object?',
            1: 'Is the {} object closer to the bottom of the image?',
            2: 'Is the {} object closer to the left side of the image?',
            3: 'What is the shape of the object that is nearest to the {} object?',
            4: 'What is the shape of the object that is farthest from the {} object?',
            5: 'What is the color of the object that is nearest to the {} object?',
            6: 'What is the color of the object that is farthest from the {} object?',
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
        color = np.argmax(encoded_question[:NUM_COLORS])
        question_code = np.argmax(encoded_question[NUM_COLORS:])
        # Return the question as a string.
        return (self.question_type_template(question_code)).format(self.color2str(color))


    def answer2str(self, encoded_answer):
        """ Encodes answer into a string.

        :param encoded_answer: One-hot vector.
        """
        return {
            # 0-5 colors
            0: 'blue',
            1: 'green',
            2: 'red',
            3: 'yellow',
            4: 'magenta',
            5: 'cyan',
            # 6-7 shapes
            6: 'rectangle',
            7: 'circle',
            # 8-9 yes/no
            8: 'yes',
            9: 'no',
        }[np.argmax(encoded_answer)]

    def scene2str(self, objects):
        """
        Returns a string with shape, color and position of every object forming the scene
        
        :param objects: List of objects - abstract scene representation.
        """
        desc = '| '
        for obj in objects:
            # Add description
            desc = desc + ('{} {} at ({}, {}) | '.format(self.color2str(obj.color), self.shape2str(obj.shape), obj.x, obj.y))
        return desc

    def generate_scene_representation(self):
        """ Generates scene representation
        
        :return: List of objects - abstract scene representation.
         """
         # Generate list of objects - no more then colors
        num_objects = np.random.random_integers(2, NUM_COLORS)

        # Shuffle "grid positions".
        grid_positions = np.arange(N_GRID*N_GRID)
        np.random.shuffle(grid_positions)
        # Size of a "grid block".
        block_size = int(self.img_size*0.9/N_GRID)

        # Shuffle colors.
        colors = np.arange(NUM_COLORS)
        np.random.shuffle(colors)
        colors = colors[:num_objects]

        # Generate shapes.
        shapes = np.random.rand(num_objects) < 0.5

        # List of objects presents in the scene.
        objects = []

        # Generate coordinates.
        for i in range(num_objects):
            # Calculate object positions depending on "grid positions"
            x = grid_positions[i] % N_GRID
            y = (N_GRID - np.floor(grid_positions[i] / N_GRID) - 1).astype(np.uint8)
            # Calculate "image coordinates".
            x_img = (x+0.5)*block_size + np.random.random_integers(-2,2)
            y_img = (y+0.5)*block_size + np.random.random_integers(-2,2)
            # Add object to list.
            objects.append(ObjectRepresentation(x_img, y_img, colors[i], shapes[i] ))

        return objects
    


    def generate_image(self, objects):
        """
        Generates image on the basis of a given scene representation 

        :param objects: List of objects - abstract scene representation.
        """
        img_size = self.img_size
        shape_size = int((img_size*0.9/N_GRID)*0.7/2)

        # Generate image [img_size, img_size, 3]
        img = Image.new('RGB', (img_size, img_size), color=BG_COLOR)
        drawer = ImageDraw.Draw(img)

        for obj in objects:
            # Calculate object position.
            position = (obj.x-shape_size, obj.y-shape_size, obj.x+shape_size, obj.y+shape_size)
            # Draw object.
            if obj.shape == 1:
                drawer.ellipse(position, fill=COLOR[obj.color])
            else:
                drawer.rectangle(position, fill=COLOR[obj.color])            

        # Cast to np.
        return np.array(img)

    def generate_question_matrix(self, objects):
        """
        Generates questions matrix: [# of shape * # of Q, # of color + # of Q]

        :param objects: List of objects - abstract scene representation.
        """
        Q = np.zeros((len(objects)*NUM_QUESTIONS, NUM_COLORS+NUM_QUESTIONS), dtype=np.bool)

        for i,obj in enumerate(objects):
            v = np.zeros(NUM_COLORS)
            v[obj.color] = True
            Q[i*NUM_QUESTIONS:(i+1)*NUM_QUESTIONS, :NUM_COLORS] = np.tile(v, (NUM_QUESTIONS, 1))
            Q[i*NUM_QUESTIONS:(i+1)*NUM_QUESTIONS, NUM_COLORS:] = np.diag(np.ones(NUM_QUESTIONS))

        return Q

    def generate_answer_matrix(self, objects):
        """
        Generates answers matrix: [# of shape * # of Q, # of color + 4]
        # of color + 4: [color 1, color 2, ... , circle, rectangle, yes, no]

        :param objects: List of objects - abstract scene representation.
        """
        A = np.zeros((len(objects)*NUM_QUESTIONS, NUM_COLORS+4), dtype=np.bool)
        for i,obj in enumerate(objects):
            # Q1: circle or rectangle?
            if obj.shape:
                A[i*NUM_QUESTIONS, NUM_COLORS+1] = True
            else:
                A[i*NUM_QUESTIONS, NUM_COLORS] = True

            # Q2: bottom?
            if obj.y > int(self.img_size/2):
                A[i*NUM_QUESTIONS+1, NUM_COLORS+2] = True
            else:
                A[i*NUM_QUESTIONS+1, NUM_COLORS+3] = True

            # Q3: left?
            if obj.x < int(self.img_size/2):
                A[i*NUM_QUESTIONS+2, NUM_COLORS+2] = True
            else:
                A[i*NUM_QUESTIONS+2, NUM_COLORS+3] = True

            # Calculate distances.
            distances = np.array([ ((obj.x - other_obj.x) ** 2 + (obj.y - other_obj.y) ** 2) for other_obj in objects])
            idx = distances.argsort()
            # Ids of closest and most distant objects. 
            min_idx = idx[1]
            max_idx = idx[-1]

            # Q4: the shape of the nearest object
            A[i*NUM_QUESTIONS+3, NUM_COLORS+objects[min_idx].shape] = True
            # Q5: the shape of the farthest object
            A[i*NUM_QUESTIONS+4, NUM_COLORS+objects[max_idx].shape] = True

            # Q6: the color of the nearest object
            A[i*NUM_QUESTIONS+5, objects[min_idx].color] = True
            # Q7: the color of the farthest object
            A[i*NUM_QUESTIONS+6, objects[max_idx].color] = True

        return A

    def generate_h5py_dataset(self):
        """
        Generates a whole new Sort-of-CLEVR dataset and saves it in the form of a HDF5 file.
        """

        # Output file.
        f = h5py.File(self.pathfilename, 'w')

        # progress bar
        bar = progressbar.ProgressBar(maxval=100, 
            widgets=[progressbar.Bar('=', '[', ']'), ' ',
            progressbar.Percentage()])
        bar.start()

        count = 0

        while(count < self.dataset_size):
            # Generate the scene.
            objects = self.generate_scene_representation()
            # Generate corresponding image, questions and answers.
            I = self.generate_image(objects)
            Q = self.generate_question_matrix(objects)
            A = self.generate_answer_matrix(objects)
            # Iterate through all questions generated for a given scene.
            for j in range(len(objects)*NUM_QUESTIONS):
                # Create new group.
                id = '{}'.format(count)
                grp = f.create_group(id)

                # Set data.
                grp['image'] = I
                grp['question'] = Q[j, :]
                grp['answer'] = A[j, :]
                grp['scene_description'] = self.scene2str(objects)

                # Increment counter.
                count += 1

                # Update progress bar.
                if count % (self.dataset_size / 100) == 0:
                    bar.update(count / (self.dataset_size / 100))
                # Check whether we generated the required number of samples - break the internal loop.
                if count >= self.dataset_size:
                    break

        # Finalize the generation.
        bar.finish()
        f.close()
        logger.info('Generated dataset with {} samples and saved to {}'.format(self.dataset_size, self.pathfilename))


    def show_sample(self, data_tuple, aux_tuple, sample_number = 0):
        """ 
        Shows a sample from the batch.

        :param data_tuple: Tuple containing inputs and targets.
        :param aux_tuple: Auxiliary tuple containing scene descriptions.
        :param sample_number: Number of sample in batch (DEFAULT: 0) 
        """
        import matplotlib.pyplot as plt

        # Unpack tuples.
        (images, questions), answers = data_tuple
        scene_descriptions = aux_tuple.scene_descriptions

        # Get sample.
        image = images[sample_number]
        question = questions[sample_number]
        answer = answers[sample_number]

        # Print scene description.
        logger.info("Scene description :\n {}".format(scene_descriptions[sample_number]))
        logger.info("Question :\n {}".format(question))
        logger.info("Answer :\n {}".format(answer))

        # Generate figure.
        fig = plt.figure(1)
        plt.title('Q: {}'.format(self.question2str(question)))
        plt.xlabel('A: {}'.format(self.answer2str(answer)))
        plt.imshow(image, interpolation='nearest', aspect='auto')
        # Plot!
        plt.show()




if __name__ == "__main__":
    """ Tests sort of CLEVR - generates and displays a sample"""

    # "Loaded parameters".
    params = {'batch_size': 100, 
        'data_folder': '~/data/sort-of-clevr/', 'data_filename': 'training.hy', 
        #'shuffle': False,
        #"regenerate": True,
        'dataset_size': 10000, 'img_size': 128
        }

    # Configure logger.
    logging.basicConfig(level=logging.DEBUG)
    logger.debug("params: {}".format(params)) 

    # Create problem object.
    problem = SortOfCLEVR(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    data_tuple, aux_tuple = next(generator)
    for i in range(params['batch_size']):
        # Display single sample from batch.
        problem.show_sample(data_tuple, aux_tuple, i)
