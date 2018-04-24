#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from problems.generate_serial_recall import generate_serial_recall
from problems.generator_scratch_pad import generator_scratch_pad
from problems.generator_ignore_distraction import generator_ignore_distraction
from problems.generate_forget_distraction import generate_forget_distraction
"""problem_factory.py: Factory building problems"""
__author__ = "Tomasz Kornuta"

class ProblemFactory(object):
    """   
    Class returning concrete models depending on the name provided in the list of parameters.
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
        name = params['name']
        if name == 'serial_recall':
            return generate_serial_recall(params)
        elif name == 'scratch_pad':
            return generator_scratch_pad(params)
        elif name == 'forget_distraction':
            return generate_forget_distraction(params)
        elif name == 'ignore_distraction':
            return generator_ignore_distraction(params)
        else:
            raise ValueError

