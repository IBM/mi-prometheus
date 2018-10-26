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
sampler_factory.py: Factory building samplers used by PyTorch's DataLoaders.

"""
__author__ = "Tomasz Kornuta"

import os
import inspect
import logging
import torch.utils.data.sampler


class SamplerFactory(object):
    """
    Class returning sampler depending on the name provided in the \
    list of parameters.
    """

    @staticmethod
    def build(problem, params):
        """
        Static method returning particular sampler, depending on the name \
        provided in the list of parameters.

        :param problem: Instance of an object derived from the Problem class.
        :type problem: ``problems.Problem``

        :param params: Parameters used to instantiate the sampler.
        :type params: ``utils.param_interface.ParamInterface``

        ..note::

            ``params`` should contains the exact (case-sensitive) class name of the sampler to instantiate.

        :return: Instance of a given sampler or None if section not present (None) or coudn't build the sampler.
        """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('SamplerFactory')

        # Check if sampler is required, i.e. 'sampler' section is empty.
        if not params:
            return None

        try: 
            # Check presence of the name attribute.
            if 'name' not in params:
                raise Exception("Sampler configuration section does not contain the key 'name'")

            # Get the class name.
            name = params['name']

            # Verify that the specified class is in the controller package.
            if name not in dir(torch.utils.data.sampler):
                raise Exception("Could not find the specified class '{}' in the controllers package".format(name))

            # Get the actual class.
            sampler_class = getattr(torch.utils.data.sampler, name)

            # Ok, proceed.
            logger.info('Loading the {} controller from {}'.format(name, sampler_class.__module__))

            # Handle "special" case.
            if name == 'SubsetRandomSampler':
                indices  = params['indices']
                # Analyze the type.
                if type(indices) == str:
                    # If indices are file - relative path.
                    filename = indices
                    if indices[0] == '~':
                        # Turn to absolute path.
                        filename = os.path.expanduser(filename)
                    # Try to open the file.
                    try: 
                        file = open(filename, "r")
                        # Read the file.
                        indices = file.readline() 
                        # Truncate the last "enter"
                        indices = indices[:-1]
                        file.close()	
                    except Exception:
                        # Ok, this is not a file.
                        pass

                    # If indices are already square brackets [].
                    if indices[0] == '[' and indices[-1] == ']':
                        # Remove the brackets.
                        indices = indices.replace("[", "").replace("]", "")

                    # Get digits.
                    digits = indices.split()
                    if len(digits) == 2:
                        # Creat simple range.
                        indices = range( int(digits[0]), int(digits[1]) )
                    else:
                        # Use them as a list.
                        indices = [int(x) for x in digits]

                # Check if there aren't too many indices.
                #if len(indices) > len(problem):
                #    logger.error("Length of indices if greater than the number of samples in the problem!")
                #    exit(-1)
                # Check if indices are within range.
                if max(indices) >= len(problem):
                    logger.error("SubsetRandomSampler cannot work properly when indices are out of range ({}) " \
                        "considering that there are {} samples in the problem!".format(max(indices), len(problem)))
                    exit(-2)
                sampler = sampler_class(indices)
            else:
                # Create "regular" sampler.
                sampler = sampler_class(problem)

            # Return sampler.
            return sampler
        except Exception as e:
            logger.error(e)
            logger.warning("Using default sampling without sampler.")
            return None


if __name__ == "__main__":
    """
    Tests the factory.
    """
    from miprometheus.utils.param_interface import ParamInterface

    # Problem.
    class TestProblem(object):
        def __len__(self):
            return 50
    # All samplers operate on TestProblem only, whereas SubsetRandomSampler additinally accepts 'incidces' with three options.
    # Option 1: range.
    indices = range(20)
    # Option 2: range as str.
    range_str = '[0 10]'
    # Option 3: list of indices.
    range_str2 = '[0 2 5 10]'
    # Option 4: name of the file containing indices.
    filename = "~/data/mnist/training_indices.txt"

    params = ParamInterface()
    params.add_default_params({'name': 'SubsetRandomSampler',
                                'indices': filename})

    sampler = SamplerFactory.build(TestProblem(), params)
    print(type(sampler))

    for i, index in enumerate(sampler):
        print('{}: index {}'.format(i, index))
        pass

