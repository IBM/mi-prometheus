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
            logger.error("Model dictionary does not contain the 'name' parameter")
            raise ValueError

        # Try to load model
        name = params['name']
        if name == 'dnc':
            logger.info('Loading the DNC model from models.dnc.dnc_model')
            sys.path.append(os.path.join(os.path.dirname(__file__),  'dnc'))
            from models.dnc.dnc_model import DNC
            return DNC(params)
        elif name == 'dwm':
            logger.info('Loading the DWM model from models.dwm.dwm_model')
            from models.dwm.dwm_model import DWM
            return DWM(params)
        elif name == 'es_lstm':
            logger.info('Loading the EncoderSolverLSTM model from models.encoder_solver.es_lstm_model')
            sys.path.append(os.path.join(os.path.dirname(__file__),  'encoder_solver'))
            from models.encoder_solver.es_lstm_model import EncoderSolverLSTM
            return EncoderSolverLSTM(params)
        elif name == 'es_ntm':
            logger.info('Loading the EncoderSolverNTM model from models.encoder_solver.es_ntm_model')
            sys.path.append(os.path.join(os.path.dirname(__file__),  'encoder_solver'))
            from models.encoder_solver.es_ntm_model import EncoderSolverNTM
            return EncoderSolverNTM(params)
        elif name == 'lstm':
            logger.info('Loading the LSTM model from models.lstm.lstm_model')
            from models.lstm.lstm_model import LSTM
            return LSTM(params)
        elif name == 'maes':
            logger.info('Loading the MAES model from models.encoder_solver.maes_model')
            sys.path.append(os.path.join(os.path.dirname(__file__),  'encoder_solver'))
            from models.encoder_solver.maes_model import MAES
            return MAES(params)
        elif name == 'mae2s':
            logger.info('Loading the MAE2S model from models.encoder_solver.mae2s_model')
            sys.path.append(os.path.join(os.path.dirname(__file__),  'encoder_solver'))
            from models.encoder_solver.mae2s_model import MAE2S
            return MAE2S(params)
        elif name == 'ntm':
            logger.info('Loading the NTM model from models.ntm.ntm_model')
            sys.path.append(os.path.join(os.path.dirname(__file__),  'ntm'))
            from models.ntm.ntm_model import NTM
            return NTM(params)
        elif name == 'seq2seqlstm':
            logger.info('Loading the EncoderDecoderLSTM model from models.seq2seqlstm.encoder_decoder_lstm')
            sys.path.append(os.path.join(os.path.dirname(__file__), 'seq2seqlstm'))
            from models.seq2seqlstm.encoder_decoder_lstm import EncoderDecoderLSTM
            return EncoderDecoderLSTM(params)
        elif name == 'simple_cnn':
            logger.info('Loading the SimpleConvNet model from models.simple_cnn.simple_cnn')
            from models.simple_cnn.simple_cnn import SimpleConvNet
            return SimpleConvNet(params)
        elif name == 'thalnet':
            logger.info('Loading the ThalNetModel model from models.thalnet.thalnet_model')
            from models.thalnet.thalnet_model import ThalNetModel
            return ThalNetModel(params)
        elif name == 'alexnet':
            logger.info('Loading the AlexnetWrapper model from models.vision.alexnet_wrapper')
            logger.warning("Warning: AlexnetWrapper not tested!")
            from models.vision.alexnet_wrapper import AlexnetWrapper
            return AlexnetWrapper(params)
        else:
            raise ValueError
