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

__author__ = "Tomasz Kornuta"

import os.path
import logging
import inspect

#from miprometheus import models


class EncoderFactory(object):
    """
    EncoderFactory: Class instantiating the specified encoder class using the passed params.
    """

    @staticmethod
    def build(params, default_values={}):
        """
        Static method returning a list of the encoders, depending on the names \
        provided in the list of parameters.

        :param params: Parameters used to instantiate the encoders.
        :type params: ``utils.param_interface.ParamInterface``

        ..note::

            ``params`` should contains the exact (case-sensitive) section "encoders" with list of class name(s) of encoders to instantiate.

        :param default_values: Default (hardcoded) values coming from a Problem class. Can be used to pass \
        values such as a number of classes, an embedding dimension etc.
        :type default_values: dict


        :return: Instance of a given encoder.

        """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('EncoderFactory')

        # Empty list.
        if "encoders" not in params:
            return []
        
        encoders = []
        # Iterate through list.
        for encoder_params in params['encoders']:
            print (encoder_params)
            # Check presence of the name attribute.
            if 'name' not in encoder_params:
                logger.error("Encoder configuration section does not contain the key 'name'")
                exit(-1)

            # Get the class name.
            name = os.path.basename(encoder_params['name'])

            # Ok, proceed.
            logger.info('Loading the {} model'.format(name))
            #logger.info('Loading the {} encoder from {}'.format(name, model_class.__module__))
            #encoder = encoder_class(params, default_values)

        # return the list
        return encoders


if __name__ == "__main__":
    """
    Tests Encoder Factory.
    """
    from miprometheus.utils.param_interface import ParamInterface
    params = ParamInterface()
    params.add_default_params({
        'encoders' : [ # LIST!
            {
                'name': 'Encoder1'
            },{
                'name': 'Encoder2'
            }
            ]
        })

    encoders = EncoderFactory.build(params)
    for encoder in encoders:
        print(type(encoder))
