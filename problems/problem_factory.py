#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from problems.serial_recall_simplified import SerialRecallSimplifiedProblem
from problems.serial_recall_original import SerialRecallOriginalProblem
from problems.scratch_pad import GeneratorScratchPad
from problems.ignore_distraction import GeneratorIgnoreDistraction
from problems.forget_distraction import GenerateForgetDistraction
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
        if name == 'serial_recall_simplified':
            return SerialRecallSimplifiedProblem(params)
        if name == 'serial_recall_original':
            return SerialRecallOriginalProblem(params)
        elif name == 'scratch_pad':
            return GeneratorScratchPad(params)
        elif name == 'forget_distraction':
            return GeneratorIgnoreDistraction(params)
        elif name == 'ignore_distraction':
            return GenerateForgetDistraction(params)
        else:
            raise ValueError

