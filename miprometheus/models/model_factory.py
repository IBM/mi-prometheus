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
import inspect

from miprometheus import models


class ModelFactory(object):
    """
    ModelFactory: Class instantiating the specified model class using the passed params.
    """

    @staticmethod
    def build(params, problem_default_values_={}):
        """
        Static method returning a particular model, depending on the name \
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
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('ModelFactory')

        # Check presence of the name attribute.
        if 'name' not in params:
            logger.error("Model configuration section does not contain the key 'name'")
            exit(-1)

        # Get the class name.
        name = os.path.basename(params['name'])

        # Verify that the specified class is in the models package.
        if name not in dir(models):
            logger.error("Could not find the specified class '{}' in the models package.".format(name))
            exit(-1)

        # Get the actual class.
        model_class = getattr(models, name)

        # Check if class is derived (even indirectly) from Model.
        inherits = False
        for c in inspect.getmro(model_class):
            if c.__name__ == models.Model.__name__:
                inherits = True
                break
        if not inherits:
            logger.error("The specified class '{}' is not derived from the Model class".format(name))
            exit(-1)

        # Ok, proceed.
        logger.info('Loading the {} model from {}'.format(name, model_class.__module__))
        # return the instantiated model class
        return model_class(params, problem_default_values_)


if __name__ == "__main__":
    """
    Tests ModelFactory.
    """
    from miprometheus.utils.param_interface import ParamInterface

    model_params = ParamInterface()
    model_params.add_default_params({'name': 'MAES',
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

    model = ModelFactory.build_model(model_params, {})
    print(type(model))
