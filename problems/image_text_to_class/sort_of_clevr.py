#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sort_of_clevr.py: Sort-of-CLEVR is a simplified version of CLEVR VQA problem """
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
from problems.image_text_to_class.image_text_to_class_problem import ImageTextToClassProblem, ImageTextTuple, SceneDescriptionTuple, ObjectRepresentation


class SortOfCLEVR(ImageTextToClassProblem):
    """
    Sort-of-CLEVR is a simple VQA problem, where the goal is to answer the question regarding a given image.
    Implementation of the generation is inspired by:
    git@github.com:gitlimlab/Relation-Network-Tensorflow.git
    Improvements:
    - generates scenes with dynamic varying number of objects (2-6)
    - more types of intra- and inter-relational questions
    - more natural interpretation of questions
    Additionally it generates:
    - Aux tuple containing the scene graph.
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
        self.regenerate = params["regenerate"]

        # training, testing data is 90%, 10% of the total data size respectively
        self.use_train_data = params['use_train_data']
        self.data_test_size = int(self.dataset_size * 0.1)
        self.data_train_size = int(self.dataset_size * 0.9)

        # Shuffle indices.
        self.shuffle = params.get('shuffle', True)

        # Set general color properties.
        self.BG_COLOR = (180, 180, 150)
        self.COLOR = [
            (0, 0, 210),    # 'blue'
            (0, 210, 0),    # 'green'
            (210, 0, 0),    # 'red'
            (150, 150, 0),  # 'yellow'
            (150, 0, 150),  # 'magenta'
            (0, 150, 150),  # 'cyan'
            # add more colors here if needed
        ]

        # Other "hyperparameters".
        self.NUM_SHAPES = 2
        self.NUM_COLORS = len(self.COLOR)
        self.NUM_QUESTIONS = 7
        # Objects are characterised by colors, so cannot have more objects than colors.
        self.MAX_NUM_OBJECTS = min(6, self.NUM_COLORS)
        self.GRID_SIZE = 4

        # Get path
        data_folder = os.path.expanduser(params['data_folder'])
        data_filename = params['data_filename']
        
        # Load or generate the dataset.
        self.load_dataset(data_folder, data_filename)

    def load_dataset(self, data_folder, data_filename):
        """ Loads the dataset from the HDF5-encoded file. If file does not exists it generates new dataset and stores it in a file. """

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
            if self.regenerate:
                raise Exception("Must regenerate... must regenerate...")
            self.data = h5py.File(self.pathfilename, 'r')
        except:
            logger.warning('File {} in {} not found. Generating new file... '.format(data_filename, data_folder))
            # Create folder - if required.
            if not os.path.exists(os.path.expanduser(data_folder)):
                os.mkdir(data_folder)
                
            # Generate the dataset, if not exists.
            # If it exists, simply load it.
            self.generate_h5py_dataset()

            # Load the file.
            self.data = h5py.File(self.pathfilename, 'r')

        logger.info("Loaded {} samples from file {}".format(len(self.data), self.pathfilename))
        
        # Generate list of indices (strings).
        if self.use_train_data:
            self.ids = ['{}'.format(i) for i in range(self.data_train_size)]
        else:
            self.ids = ['{}'.format(i) for i in range(self.data_train_size, self.data_train_size+self.data_test_size)]

    def generate_batch(self):
        """ Generates batch.
        
        :return: DataTuple and AuxTuple object.
        """
        # Shuffle indices.
        if self.shuffle:
            random.shuffle(self.ids)
        # Get batch of indices.
        batch_ids = self.ids[:self.batch_size]

        # Get batch.
        images = []
        questions = []
        answers = []
        scenes = []

        for id in batch_ids:
            group = self.data[id]

            # Process data
            images.append((group['image'].value/255).transpose(2,1,0))
            questions.append(group['question'].value.astype(np.float32)) 
            answers.append(group['answer'].value.astype(np.float32)) 
            scenes.append(group['scene_description'].value)

        # Generate tuple with inputs
        inputs = ImageTextTuple(torch.from_numpy(np.stack(images, axis=0)).type(torch.FloatTensor), torch.from_numpy(np.stack(questions, axis=0)))
        targets = np.stack(answers, 0)
        index_targets = torch.from_numpy(np.argmax(targets, axis=1))

        # Add scene decription to aux tuple.
        aux_tuple = SceneDescriptionTuple(scenes)

        # Return DataTuple(!) and an AuxTuple with scene description.
        return DataTuple(inputs, index_targets), aux_tuple

    def color2str(self, color_code):
        " Decodes color and returns it as a string. "
        return {
            0: 'blue',
            1: 'green',
            2: 'red',
            3: 'yellow',
            4: 'magenta',
            5: 'cyan',
        }[int(color_code)]

    def shape2str(self, shape_code):
        " Decodes shape and returns it as a string. "
        return {
            0: 'rectangle',
            1: 'circle',
        }[int(shape_code)]

    def question_type_template(self, question_code):
        """ Helper function that the string templates a question type. """
        return {
            0: 'What is the shape of the {} object?',
            1: 'Is the {} {} closer to the bottom of the image?',
            2: 'Is the {} {} closer to the left side of the image?',
            3: 'What is the shape of the object nearest to the {} {}?',
            4: 'What is the shape of the object farthest from the {} {}?',
            5: 'What is the color of the object nearest to the {} {}?',
            6: 'What is the color of the object farthest from the {} {}?',
            #7: 'How many objects have the same shape as the {} {}?,
        }[int(question_code)]

    def question2str(self, encoded_question):
        """ Decodes question, i.e. produces a human-understandable string. 
        
        :param color_query: Concatenation of two one-hot vectors: 
          - first one denoting the object of interest (its color), 
          - the second one the question type.
        :return: Question in the form of a string.
        """
        # "Decode" the color_query vector.
        color = np.argmax(encoded_question[:self.NUM_COLORS])
        question_code = np.argmax(encoded_question[self.NUM_COLORS:])
        # Return the question as a string.
        return (self.question_type_template(question_code)).format(self.color2str(color), 'object')

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
        }[int(encoded_answer)]

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
         # Generate list of objects - no more then colors.
        num_objects = np.random.random_integers(2, self.MAX_NUM_OBJECTS)

        # Shuffle "grid positions".
        grid_positions = np.arange(self.GRID_SIZE*self.GRID_SIZE)
        np.random.shuffle(grid_positions)
        # Size of a "grid block".
        block_size = int(self.img_size*0.9/self.GRID_SIZE)

        # Shuffle colors.
        colors = np.arange(self.NUM_COLORS)
        np.random.shuffle(colors)
        colors = colors[:num_objects]

        # Generate shapes.
        shapes = (np.random.rand(num_objects) < 0.5).astype(int)

        # List of objects presents in the scene.
        objects = []

        # Generate coordinates.
        for i in range(num_objects):
            # Calculate object positions depending on "grid positions"
            x = grid_positions[i] % self.GRID_SIZE
            y = (self.GRID_SIZE - np.floor(grid_positions[i] / self.GRID_SIZE) - 1).astype(np.uint8)
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
        shape_size = int((img_size*0.9/self.GRID_SIZE)*0.7/2)

        # Generate image [img_size, img_size, 3]
        img = Image.new('RGB', (img_size, img_size), color=self.BG_COLOR)
        drawer = ImageDraw.Draw(img)

        for obj in objects:
            # Calculate object position.
            position = (obj.x-shape_size, obj.y-shape_size, obj.x+shape_size, obj.y+shape_size)
            # Draw object.
            if obj.shape == 1:
                drawer.ellipse(position, fill=self.COLOR[obj.color])
            else:
                drawer.rectangle(position, fill=self.COLOR[obj.color])            

        # Cast to np.
        return np.array(img)

    def generate_question_matrix(self, objects):
        """
        Generates questions matrix: [# of shape * # of Q, # of color + # of Q]

        :param objects: List of objects - abstract scene representation.
        """
        Q = np.zeros((len(objects)*self.NUM_QUESTIONS, self.NUM_COLORS+self.NUM_QUESTIONS), dtype=np.bool)

        for i,obj in enumerate(objects):
            v = np.zeros(self.NUM_COLORS)
            v[obj.color] = True
            Q[i*self.NUM_QUESTIONS:(i+1)*self.NUM_QUESTIONS, :self.NUM_COLORS] = np.tile(v, (self.NUM_QUESTIONS, 1))
            Q[i*self.NUM_QUESTIONS:(i+1)*self.NUM_QUESTIONS, self.NUM_COLORS:] = np.diag(np.ones(self.NUM_QUESTIONS))

        return Q

    def generate_answer_matrix(self, objects):
        """
        Generates answers matrix: [# of shape * # of Q, # of color + 4]
        # of color + 4: [color 1, color 2, ... , circle, rectangle, yes, no]

        :param objects: List of objects - abstract scene representation.
        """
        A = np.zeros((len(objects)*self.NUM_QUESTIONS, self.NUM_COLORS+4), dtype=np.bool)
        for i,obj in enumerate(objects):
            # Q1: circle or rectangle?
            if obj.shape:
                A[i*self.NUM_QUESTIONS, self.NUM_COLORS+1] = True
            else:
                A[i*self.NUM_QUESTIONS, self.NUM_COLORS] = True

            # Q2: bottom?
            if obj.y > int(self.img_size/2):
                A[i*self.NUM_QUESTIONS+1, self.NUM_COLORS+2] = True
            else:
                A[i*self.NUM_QUESTIONS+1, self.NUM_COLORS+3] = True

            # Q3: left?
            if obj.x < int(self.img_size/2):
                A[i*self.NUM_QUESTIONS+2, self.NUM_COLORS+2] = True
            else:
                A[i*self.NUM_QUESTIONS+2, self.NUM_COLORS+3] = True

            # Calculate distances.
            distances = np.array([ ((obj.x - other_obj.x) ** 2 + (obj.y - other_obj.y) ** 2) for other_obj in objects])
            idx = distances.argsort()
            # Ids of closest and most distant objects. 
            min_idx = idx[1]
            max_idx = idx[-1]

            # Q4: the shape of the nearest object
            A[i*self.NUM_QUESTIONS+3, self.NUM_COLORS+objects[min_idx].shape] = True
            # Q5: the shape of the farthest object
            A[i*self.NUM_QUESTIONS+4, self.NUM_COLORS+objects[max_idx].shape] = True

            # Q6: the color of the nearest object
            A[i*self.NUM_QUESTIONS+5, objects[min_idx].color] = True
            # Q7: the color of the farthest object
            A[i*self.NUM_QUESTIONS+6, objects[max_idx].color] = True

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
            for j in range(len(objects)*self.NUM_QUESTIONS):
                # Create new group.
                id = '{}'.format(count)
                grp = f.create_group(id)

                # Set data.
                grp['image'] = I
                grp['question'] = Q[j, ...]
                grp['answer'] = A[j, ...]
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
        image = images[sample_number].numpy().transpose(2, 1, 0)
        question = questions[sample_number]
        answer = answers[sample_number]

        # Print scene description.
        logger.info("Scene description :\n {}".format(scene_descriptions[sample_number]))
        logger.info("Question :\n {} ({})".format(question, self.question2str(question)))
        logger.info("Answer :\n {} ({})".format(answer, self.answer2str(answer)))

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
    params = {'batch_size': 10,
        'data_folder': '~/data/sort-of-clevr/', 'data_filename': 'training.hy',
        'use_train_data':False,
        #'shuffle': False,
        #"regenerate": True,
        'dataset_size': 10000, 'img_size': 128, 'regenerate': False
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
