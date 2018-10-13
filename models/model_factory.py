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
model_factory.py: Utility constructing a model class using specified parameters.

"""
__author__ = "Tomasz Kornuta & Vincent Marois"

import os.path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ModelFactory')


class ModelFactory(object):
    """
    ModelFactory: Class instantiating the specified model class using the passed params.
    """

    @staticmethod
    def build_model(params, problem_default_values_={}):
        """
        Static method returning a particular model, depending on the name\
        provided in the list of parameters.

        :param params: Parameters used to instantiate the model class.
        :type params: ``utils.param_interface.ParamInterface``

        ..note::

            ``params`` should contains the exact (case-sensitive) class name of the model to instantiate.

        :param problem_default_values_: Default (hardcoded) values coming from a Problem class. Can be used to pass \
        values such as a number of classes, an embedding dimension etc.
        :type problem_default_values_: dict


        :return: Instance of a given model.

        """
        # Check presence of the name
        if 'name' not in params:
            logger.error("Model parameter dictionary does not contain the key 'name'.")
            raise KeyError

        # get the class name
        name = os.path.basename(params['name'])

        # import the models package
        import models

        # verify that the specified class is in the models package
        if name not in dir(models):
            logger.error("Could not find the specified classname in the models package.")
            raise KeyError

        # get the actual class
        model_class = getattr(models, name)
        logger.info('Loading the {} model from {}'.format(name, model_class.__module__))

        # return the instantiated model class
        return model_class(params, problem_default_values_)


if __name__ == "__main__":
    """
    Tests ModelFactory.
    """
    from problems.problem_factory import ProblemFactory
    from utils.param_interface import ParamInterface

    problem_params = ParamInterface()
    problem_params.add_custom_params({'name': 'SerialRecallCommandLines',
                                      'control_bits': 4,
                                      'data_bits': 8,
                                      'min_sequence_length': 2,
                                      'max_sequence_length': 5})

    # create problem
    problem = ProblemFactory.build_problem(problem_params)

    model_params = ParamInterface()
    model_params.add_custom_params({'name': 'MAES',
                                    'num_control_bits': 3,
                                    'num_data_bits': 8,
                                    'encoding_bit': 0,
                                    'solving_bit': 1,
                                    'controller': {'name': 'rnn', 'hidden_state_size': 20,
                                                   'num_layers': 1, 'non_linearity': 'sigmoid'},
                                    'mae_interface': {'shift_size': 3},
                                    'mas_interface': {'shift_size': 3},
                                    'memory': {'num_addresses': -1, 'num_content_bits': 11},
                                    'visualization_mode': 2})

    model = ModelFactory.build_model(model_params, problem.default_values)
    print(type(model))
