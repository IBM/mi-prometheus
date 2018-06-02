#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""problem_factory.py: Factory building problems"""
__author__ = "Tomasz Kornuta"

import sys, inspect
import os.path
import glob


class ProblemFactory(object):
    """
    Class returning concrete problem/generator depending on the name provided in the list of parameters.
    """

    @staticmethod
    def build_problem(params):
        """ Static method returning particular problem, depending on the name provided in the list of parameters.

        :param params: Dictionary of parameters (in particular containing 'name' which is equivalend to problem name)
        :returns: Instance of a given problem.
        """
        # Check name
        if 'name' not in params:
            print("Problem parameter dictionary does not contain 'name'")
            raise ValueError
        # Try to load model
        name = os.path.basename(params['name'])

        # search recursively the name of the problem
        for filename in glob.iglob('problems/**/*.py', recursive=True):
            if os.path.basename(filename) == name + '.py':
                # append system path
                sys.path.append(os.path.dirname(filename))

        # Import module
        module = __import__(name)
        # Get classes from that module.
        is_class_member = lambda member: inspect.isclass(member) and member.__module__ == name
        clsmembers = inspect.getmembers(sys.modules[name], is_class_member)
        # Assert there is only one class.
        assert len(clsmembers) == 1
        class_name = clsmembers[0][0]
        # Get problem class
        problem_class = getattr(module, class_name)
        print('Successfully loaded problem {} from {}'.format(class_name, name))
        # Create problem object.
        return problem_class(params)


if __name__ == "__main__":
    """ Tests problem factory"""
    # Problem name
    params = {'name': 'serial_recall', 'control_bits': 3, 'data_bits': 8, 'batch_size': 1,
              'min_sequence_length': 1, 'max_sequence_length': 10, 'num_subseq_min': 1, 'num_subseq_max': 5,
              'bias': 0.5}

    problem = ProblemFactory.build_problem(params)
    # Get generator
    generator = problem.return_generator()
    # Get batch.
    (x, y, mask) = next(generator)
    # Display single sample (0) from batch.
    problem.show_sample(x, y, mask)


