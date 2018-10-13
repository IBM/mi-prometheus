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
problem_factory.py: Utility constructing a problem class using specified parameters.

"""
__author__ = "Tomasz Kornuta & Vincent Marois"

import os.path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ProblemFactory')


class ProblemFactory(object):
    """
    ProblemFactory: Class instantiating the specified problem class using the passed params.
    """

    @staticmethod
    def build_problem(params):
        """
        Static method returning a particular problem, depending on the name\
        provided in the list of parameters.

        :param params: Parameters used to instantiate the Problem class.
        :type params: ``utils.param_interface.ParamInterface``

        ..note::

            ``params`` should contains the exact (case-sensitive) class name of the Problem to instantiate.


        :return: Instance of a given problem.

        """
        # Check presence of the name
        if 'name' not in params:
            logger.error("Problem parameter dictionary does not contain the key 'name'.")
            raise KeyError

        # get the class name
        name = os.path.basename(params['name'])

        # import the problems package
        import problems

        # verify that the specified class is in the problems package
        if name not in dir(problems):
            logger.error("Could not find the specified classname in the problems package.")
            raise KeyError

        # get the actual class
        problem_class = getattr(problems, name)
        logger.info('Loading the {} problem from {}'.format(name, problem_class.__module__))

        # return the instantiated problem class
        return problem_class(params)


if __name__ == "__main__":
    """
    Tests ProblemFactory.
    """
    from utils.param_interface import ParamInterface
    params = ParamInterface()
    params.add_custom_params({'name': 'SerialRecall',
                              'control_bits': 3,
                              'data_bits': 8,
                              'batch_size': 1,
                              'min_sequence_length': 1,
                              'max_sequence_length': 10,
                              'num_subseq_min': 1,
                              'num_subseq_max': 5,
                              'bias': 0.5})

    problem = ProblemFactory.build_problem(params)
    print(type(problem))
