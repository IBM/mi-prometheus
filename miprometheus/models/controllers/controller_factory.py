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
controller_factory.py: Factory building controllers for MANNs.

"""
__author__ = "Ryan L. McAvoy & Vincent Marois"
import logging
import inspect
from torch.nn import Module

import miprometheus.models.controllers


class ControllerFactory(object):
    """
    Class returning concrete controller depending on the name provided in the \
    list of parameters.
    """

    @staticmethod
    def build(params):
        """
        Static method returning particular controller, depending on the name \
        provided in the list of parameters.

        :param params: Parameters used to instantiate the controller.
        :type params: ``utils.param_interface.ParamInterface``

        ..note::

            ``params`` should contains the exact (case-sensitive) class name of the controller to instantiate.

        :return: Instance of a given controller.
        """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('ControllerFactory')

        # Check presence of the name attribute.
        if 'name' not in params:
            logger.error("Controller configuration section does not contain the key 'name'")
            exit(-1)

        # Get the class name.
        name = params['name']

        # Verify that the specified class is in the controller package.
        if name not in dir(miprometheus.models.controllers):
            logger.error("Could not find the specified class '{}' in the controllers package.".format(name))
            exit(-1)

        # Get the actual class.
        controller_class = getattr(miprometheus.models.controllers, name)

        # Check if class is derived (even indirectly) from nn.Module.
        inherits = False
        for c in inspect.getmro(controller_class):
            if c.__name__ == Module.__name__:
                inherits = True
                break
        if not inherits:
            logger.error("The specified class '{}' is not derived from the nn.Module class".format(name))
            exit(-1)

        # Ok, proceed.
        logger.info('Loading the {} controller from {}'.format(name, controller_class.__module__))

        # return the instantiated controller class
        return controller_class(params)


if __name__ == "__main__":
    """
    Tests ControllerFactory.
    """
    from miprometheus.utils.param_interface import ParamInterface

    controller_params = ParamInterface()
    controller_params.add_default_params({'name': 'RNNController',
                                          'input_size': 11,
                                          'output_size': 11,
                                          'hidden_state_size': 20,
                                          'num_layers': 1,
                                          'non_linearity': 'sigmoid'})

    controller = ControllerFactory.build_controller(controller_params)
    print(type(controller))
