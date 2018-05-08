#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model_factory.py: Factory building models"""
__author__ = "Tomasz Kornuta"

class ControllerFactory(object):
    """   
    Class returning concrete models depending on the name provided in the list of parameters.
    """
    
    @staticmethod
    def build_model(params):
        """ Static method returning particular model, depending on the name provided in the list of parameters.
        
        :param params: Dictionary of parameters (in particular containing 'name' which is equivalend to model name)
        :returns: Instance of a given model.
        """
        # Check name
        if 'controller_type' not in params:
            print("Model parameter dictionary does not contain 'name'")
            raise ValueError
        # Try to load model
        name = params['type']
        if name == 'lstm':
            from models.dnc.lstm_controller import LSTMController
            return LSTMController(params)
        elif name == 'rnn':
            from models.dnc.rnn_controller import RNNController
            return RNNController(params)
        elif name == 'rnn_sig':
            from models.dnc.rnn_sigmoid_controller import RNNController
            return RNNController(params)
        elif name == 'ffn':
            from models.dnc._feedforward_controller import FeedforwardController
            return FeedforwardController(params)
        else:
            raise ValueError
