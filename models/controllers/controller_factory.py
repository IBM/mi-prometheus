#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""controller_factory.py: Factory building controllers for MANNs"""
__author__ = "Ryan L. McAvoy"


class ControllerFactory(object):
    """   
    Class returning concrete controller depending on the name provided in the list of parameters.
    """
    
    @staticmethod
    def build_model(params):
        """ Static method returning particular controller, depending on the name provided in the list of parameters.
        
        :param params: Dictionary of parameters (in particular containing 'name' which is equivalend to controller name)
        :returns: Instance of a given model.
        """
        # Check name
        if 'name' not in params:
            print("Model Controller parameter dictionary does not contain 'name'")
            raise ValueError
        # Try to load model
        name = params['name']
        if name == 'lstm':
            from lstm_controller import LSTMController
            return LSTMController(params)
        elif name == 'rnn':
            from rnn_controller import RNNController
            return RNNController(params)
        elif name == 'ffn':
            from feedforward_controller import FeedforwardController
            return FeedforwardController(params)
        elif name == 'gru':
            from gru_controller import GRUController
            return GRUController(params)
        elif name == 'ffgru':
            from ffgru_controller import FFGRUController
            return FFGRUController(params)
        else:
            raise ValueError
