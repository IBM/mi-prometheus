#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""shape_color_query.py: ShapeColorQuery is a a variation of Sort-of-CLEVR VQA problem, where question is a sequence composed of two items: 
first encoding the object type, and second encoding the query. """
__author__ = "Tomasz Kornuta"

import numpy as np

import logging
logger = logging.getLogger('Shape-Color-Query')

# Add path to main project directory - required for testing of the main function and see whether problem is working at all (!)
import os,  sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..','..')) 

from problems.image_text_to_class.sort_of_clevr import SortOfCLEVR


class ShapeColorQuery(SortOfCLEVR):
    """
    Shape-Color-Query is a a variation of Sort-of-CLEVR VQA problem, where question is a sequence composed of three items: 
    - first two encoding the object, identified by color & shape, and 
    - third encoding the query.
    """

    def __init__(self, params):
        """
        Initializes Shape-Color-Query problem, calls base class initialization, sets properties using the provided parameters.

        :param params: Dictionary of parameters (read from configuration file).
        """

        # Call base class constructors.
        super(ShapeColorQuery, self).__init__(params)


    def question2str(self, encoded_question):
        """ Decodes question, i.e. produces a human-understandable string. 
        
        :param color_query: A 3d tensor, with 1 row and 3 columns: 
        - first two encoding the object, identified by shape, color, and 
        - third encoding the query.
        :return: Question in the form of a string.
        """
        # "Decode" the question.
        if max(encoded_question[0, :]) == 0:
            shape = 'object'
        else:
            shape = self.shape2str(np.argmax(encoded_question[0, :]))
        color = self.color2str(np.argmax(encoded_question[1, :]))
        query = self.question_type_template(np.argmax(encoded_question[2, :]))
        # Return the question as a string.
        return query.format(color, shape)


    def generate_question_matrix(self, objects):
        """
        Generates questions tensor: [# of objects * # of Q, 3, encoding] 
        where second dimension ("temporal") encodes consecutivelly: shape, color, query

        :param objects: List of objects - abstract scene representation.
        :return: a 3D tensor [# of questions for the whole scene, 3, num_bits] 
        """
        # Number of scene questions.
        num_questions = len(objects)*self.NUM_QUESTIONS
        # Number of bits in Object and Query vectors.
        num_bits = max(self.NUM_COLORS, self.NUM_SHAPES, self.NUM_QUESTIONS)

        # Create query tensor.
        Q = np.zeros((num_questions, 3, num_bits), dtype=np.bool)

        # Helper matrix - queries for all question types.
        query_matrix = np.diag(np.ones(num_bits))

        # For every object in the scene.
        for i,obj in enumerate(objects):
            # Shape - with special case: query 0 asks about shape, do not provide answer as part of the query! (+1)
            Q[i*self.NUM_QUESTIONS + 1:(i+1)*self.NUM_QUESTIONS, 0, obj.shape] = True
            # Color
            Q[i*self.NUM_QUESTIONS:(i+1)*self.NUM_QUESTIONS, 1, obj.color] = True
            # Query.
            Q[i*self.NUM_QUESTIONS:(i+1)*self.NUM_QUESTIONS, 2, :num_bits] = query_matrix[:self.NUM_QUESTIONS, :num_bits]
        
        return Q


if __name__ == "__main__":
    """ Tests Shape-Color-Query - generates and displays a sample"""

    # "Loaded parameters".
    params = {'batch_size': 10,
        'data_folder': '~/data/shape-color-query/', 'data_filename': 'training.hy', 
        'shuffle': True,
        "regenerate": True,
        'dataset_size': 100, 'img_size': 224
        }

    # Configure logger.
    logging.basicConfig(level=logging.DEBUG)
    logger.debug("params: {}".format(params)) 

    # Create problem object.
    problem = ShapeColorQuery(params)

    # Get generator
    generator = problem.return_generator()

    # Get batch.
    data_tuple, aux_tuple = next(generator)
    for i in range(params['batch_size']):
        (images, texts), _ = data_tuple

        # Display single sample from batch.
        problem.show_sample(data_tuple, aux_tuple, i)
