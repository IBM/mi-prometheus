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
        if name == 'dnc':
            sys.path.append(os.path.join(os.path.dirname(__file__),  'dnc'))
            from models.dnc.dnc_model import DNC
            return DNC(params)
        elif name == 'dwm':
            from models.dwm.dwm_model import DWM
            return DWM(params)
        elif name == 'es_lstm':
            sys.path.append(os.path.join(os.path.dirname(__file__),  'encoder_solver'))
            from models.encoder_solver.es_lstm_model import EncoderSolverLSTM
            return EncoderSolverLSTM(params)
        elif name == 'lstm':
            from models.lstm.lstm_model import LSTM
            return LSTM(params)
        elif name == 'ntm':
            logger.warning("Warning: NTM not fully operational yet!")
            sys.path.append(os.path.join(os.path.dirname(__file__),  'ntm'))
            from models.ntm.ntm_model import NTM
            return NTM(params)
        elif name == 'seq2seqlstm':
            sys.path.append(os.path.join(os.path.dirname(__file__), 'seq2seqlstm'))
            from models.seq2seqlstm.encoder_decoder_lstm import EncoderDecoderLSTM
            return EncoderDecoderLSTM(params)
        elif name == 'simple_cnn':
            from models.simple_cnn.simple_cnn import SimpleConvNet
            return SimpleConvNet(params)
        elif name == 'thalnet':
            from models.thalnet.thalnet_model import ThalNetModel
            return ThalNetModel(params)
        elif name == 'alexnet':
            from models.vision.alexnet_wrapper import AlexnetWrapper
            return AlexnetWrapper(params)
        else:
            raise ValueError
