#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model_factory.py: Factory building models"""
__author__ = "Tomasz Kornuta"

import sys
import os.path
import logging
logger = logging.getLogger('ModelFactory')

class ModelFactory(object):
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
        if 'name' not in params:
            print("Model parameter dictionary does not contain 'name'")
            raise ValueError
        # Try to load model
        name = params['name']
        if name == 'dwm':
            from models.dwm.dwm_layer import DWM
            return DWM(params)
        elif name == 'lstm':
            from models.lstm.layer import LSTM
            return LSTM(params)
        elif name == 'dnc':
            sys.path.append(os.path.join(os.path.dirname(__file__),  'dnc'))
            from models.dnc.dnc_layer import DNC
            return DNC(params)
        elif name == 'ntm':
            logger.warning("Warning: NTM not fully operational yet!")
            sys.path.append(os.path.join(os.path.dirname(__file__),  'ntm'))
            from models.ntm.ntm_module import NTM
            return NTM(params)
        else:
            raise ValueError
