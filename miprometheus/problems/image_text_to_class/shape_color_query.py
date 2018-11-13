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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
shape_color_query.py: ShapeColorQuery is a a variation of the ``Sort-of-CLEVR`` problem, where the question is a\
 sequence composed of two items:

    - The first encodes the object type,
    - The second encodes the query.

"""
__author__ = "Tomasz Kornuta & Vincent Marois"
import torch
import numpy as np
from miprometheus.problems.image_text_to_class.sort_of_clevr import SortOfCLEVR


class ShapeColorQuery(SortOfCLEVR):
    """
    Shape-Color-Query is a variation of the ``Sort-of-CLEVR`` problem, where\
     the question is a sequence composed of three items:

        - The first two are encoding the object, identified by its color & shape,
        - The third is encoding the query.

    Please see the ``SortOfCLEVR`` documentation for more information.

    """

    def __init__(self, params):
        """
        Initializes the ``Shape-Color-Query`` problem, calls base class ``SortOfCLEVR``\
        initialization, sets properties using the provided parameters.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).
        :type params: miprometheus.utils.ParamInterface

        .. note::

            The following is set by default:

            >>> params = {'data_folder': '~/data/shape-color-query/',
            >>>           'split': 'train',
            >>>           'regenerate': False,
            >>>           'size': 10000,
            >>>           'img_size': 128}


        """

        # Call base class constructors.
        super(ShapeColorQuery, self).__init__(params)
        # problem name
        self.name = 'Shape-Color-Query'

        # Add default values of parameters.
        self.params.add_default_params({'data_folder': '~/data/shape-color-query/',
                                        'split': 'train',
                                        'regenerate': False,
                                        'size': 10000,
                                        'img_size': 128})

        # define the data_definitions dict: holds a description of the DataDict content
        self.data_definitions = {'images': {'size': [-1, 3, self.img_size, self.img_size], 'type': [torch.Tensor]},
                                 'questions': {'size': [-1, 3, self.NUM_QUESTIONS],
                                               'type': [torch.Tensor]},
                                 'targets_classes': {'size': [-1, self.NUM_COLORS + self.NUM_SHAPES + 2],
                                                     'type': [torch.Tensor]},
                                 'targets': {'size': [-1], 'type': [torch.Tensor]},
                                 'scenes_description': {'size': [-1, -1], 'type': [list, str]},
                                 }

        # define the default_values dict: holds parameters values that a model may need.
        self.default_values = {'height': self.img_size,
                               'width': self.img_size,
                               'num_channels': 3,
                               'num_classes': 10,
                               'question_size': 7,  # 'encoding' size of the shape, color & query
                               'seq_length': 3}  # nb of elts in the question: shape, color, query

    def question2str(self, encoded_question):
        """
        Decodes the question, i.e. produces a human-understandable string.

        :param encoded_question: A 3D tensor, with 1 row and 3 columns:

            - The first two are encoding the object, identified by its shape & color,
            - The third is encoding the query.

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
        Generates the questions tensor: [# of objects * # of Q, 3, encoding],\
        where the 2nd dimension (`temporal`) encodes consecutively: shape, color, query

        :param objects: List of objects - abstract scene representation.
        :type object: list

        :return: a 3D tensor [# of questions for the whole scene, 3, num_bits]

        """

        # Number of scene questions.
        num_questions = len(objects) * self.NUM_QUESTIONS
        # Number of bits in Object and Query vectors.
        num_bits = max(self.NUM_COLORS, self.NUM_SHAPES, self.NUM_QUESTIONS)

        # Create query tensor.
        Q = np.zeros((num_questions, 3, num_bits), dtype=np.bool)

        # Helper matrix - queries for all question types.
        query_matrix = np.diag(np.ones(num_bits))

        # For every object in the scene.
        for i, obj in enumerate(objects):
            # Shape - with special case: query 0 asks about shape, do not
            # provide answer as part of the query! (+1)
            Q[i * self.NUM_QUESTIONS +
                1:(i + 1) * self.NUM_QUESTIONS, 0, obj.shape] = True
            # Color
            Q[i * self.NUM_QUESTIONS:(i + 1) * self.NUM_QUESTIONS,
              1, obj.color] = True
            # Query.
            Q[i * self.NUM_QUESTIONS:(i + 1) * self.NUM_QUESTIONS, 2,
              :num_bits] = query_matrix[:self.NUM_QUESTIONS, :num_bits]

        return Q


if __name__ == "__main__":
    """ Tests Shape-Color-Query - generates and displays a sample"""

    # "Loaded parameters".
    from miprometheus.utils.param_interface import ParamInterface

    params = ParamInterface()  # using the default values

    # create problem
    shapecolorquery = ShapeColorQuery(params)

    batch_size = 64
    print('Number of episodes to run to cover the set once: {}'.format(shapecolorquery.get_epoch_size(batch_size)))

    # get a sample
    sample = shapecolorquery[0]
    print(repr(sample))
    print('__getitem__ works.')

    # wrap DataLoader on top of this Dataset subclass
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset=shapecolorquery, collate_fn=shapecolorquery.collate_fn,
                            batch_size=batch_size, shuffle=True, num_workers=0)

    # try to see if there is a speed up when generating batches w/ multiple workers
    import time

    s = time.time()
    for i, batch in enumerate(dataloader):
        print('Batch # {} - {}'.format(i, type(batch)))

    print('Number of workers: {}'.format(dataloader.num_workers))
    print('time taken to exhaust the dataset for a batch size of {}: {}s'.format(batch_size, time.time() - s))

    # Display single sample (0) from batch.
    batch = next(iter(dataloader))
    shapecolorquery.show_sample(batch, 0)

    print('Unit test completed')