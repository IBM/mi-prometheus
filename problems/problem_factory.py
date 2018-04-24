#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""problem_factory.py: Factory building problems"""
__author__      = "Tomasz Kornuta"

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
        if name == 'serial_recall_v1':
            return "SERIAL_RECALL_V1"
        else:
            raise ValueError

