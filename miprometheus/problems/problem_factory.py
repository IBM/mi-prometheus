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
import inspect

from miprometheus import problems


class ProblemFactory(object):
    """
    ProblemFactory: Class instantiating the specified problem class using the passed params.
    """

    @staticmethod
    def build(params):
        """
        Static method returning a particular problem, depending on the name \
        provided in the list of parameters.

        :param params: Parameters used to instantiate the Problem class.
        :type params: :py:class:`miprometheus.utils.ParamInterface`

        ..note::

            ``params`` should contains the exact (case-sensitive) class name of the Problem to instantiate.


        :return: Instance of a given problem.

        """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('ProblemFactory')

        import pkgutil
        #print([name for _, name, _ in pkgutil.iter_modules(['miprometheus.problems'])])
        #print(inspect.getmembers(problems, inspect.ismodule))
        #exit(1)


        # Check presence of the name
        if 'name' not in params:
            logger.error("Problem configuration section does not contain the key 'name'")
            exit(-1)

        # Get the class name.
        name = os.path.basename(params['name'])

        # Get the actual class.
        problem_class = getattr(problems, name)

        # Check if class is derived (even indirectly) from Problem.
        inherits = False
        for c in inspect.getmro(problem_class):
            if c.__name__ == problems.Problem.__name__:
                inherits = True
                break
        if not inherits:
            logger.error("The specified class '{}' is not derived from the Problem class".format(name))
            exit(-1)

        # Ok, proceed.
        logger.info('Loading the {} problem from {}'.format(name, problem_class.__module__))
        # return the instantiated problem class
        return problem_class(params)


if __name__ == "__main__":
    """
    Tests Problem Factory.
    """
    from miprometheus.utils.param_interface import ParamInterface
    params = ParamInterface()
    params.add_default_params({
        'name': 'SerialRecallCommandLines',
        'control_bits': 3,
        'data_bits': 8,
        'min_sequence_length': 1,
        'max_sequence_length': 5
        })

    problem = ProblemFactory.build(params)
    print(type(problem))
