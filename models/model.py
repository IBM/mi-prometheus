#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model.py: contains base abstract model for all models"""
__author__ = "Tomasz Kornuta"

import torch
from torch import nn
from abc import abstractmethod
import numpy as np

import logging

from utils.app_state import AppState


class Model(nn.Module):
    """
    Class representing base class of all models.

    Provides basic plotting functionality.

    """

    def __init__(self, params, problem_default_values_={}):
        """
        Initializes application state and sets plot if visualization flag is
        turned on.

        :param params: Parameters read from configuration file.

        :param problem_default_values_: dict of parameters values coming from the problem class. One example of such\
        parameter value is the size of the vocabulary set in a translation problem.
        :type problem_default_values_: dict

        """
        # Call base class constructor here.
        super(Model, self).__init__()

        # "Default" model name.
        self.name = 'Model'

        # initialize the logger
        self.logger = logging.getLogger(self.name)

        # process all params from configuration file and problem_default_values_ here --

        # Store pointer to params.
        self.params = params

        # Flag indicating whether intermediate checkpoints should be saved or
        # not (DEFAULT: False).
        if "save_intermediate" not in params:
            params.add_default_params({"save_intermediate": False})
        self.save_intermediate = params["save_intermediate"]

        try:
            for key in problem_default_values_.keys():
                self.params.add_custom_params({key: problem_default_values_[key]})

        except BaseException:
            self.logger.info('No parameter value was parsed from problem_default_values_')

        # --> We assume from here that the model class has all parameters values needed (either from params or
        # problem_default_values_ to correctly be instantiated) contained in self.params.

        # We can then define a dict that contains a description of the expected (and mandatory) inputs for this model.
        # This dict should be defined using self.params.
        self.data_definitions = {'inputs': {'size': [], 'type': []}}

        # --> The remaining parameters should be hardcoded values.

        # Initialize app state.
        self.app_state = AppState()

        # Window in which the data will be plotted.
        self.plotWindow = None

        # Initialization of best loss - as INF.
        self.best_loss = np.inf

    def handshake_definitions(self, problem_data_definitions_):
        """
        Proceeds to the handshake between what the Problem class provides (through a ``DataDict``) and what the model\
        expects as inputs.

        .. note::

            Handshaking is defined here as making sure that the ``Model`` and the ``Problem`` agree on the data that they
            exchange.
            More specifically, the ``Model`` has a definition of the inputs data that it expects\
            (through its ``self.data_definitions`` attribute). The ``Problem`` has the same object describing what \
            it generates.

            This functions proceeds to the handshaking as:

                - Verifying that all key existing in ``Model.data_definitions`` are also existing in \
                  ``Problem.data_definitions``. If a key is missing, an exception is thrown.
                - If all keys are present, than this function checks that for each (``Model.data_definitions``) key,\
                 the shape and type of the corresponding value matches what is indicated for the corresponding key\
                 in ``Problem.data_definitions``. If not, an exception is thrown.
                - **If both steps above passed, than the Model accepts what the Problem generates and can proceed \
                to the forward pass.**


            To properly define the ``data_definitions`` dicts, here are some examples:

                >>> data_definitions = {'img': {'size': [-1, 320, 480, 3], 'type': [np.ndarray]},
                >>>                     'question': {'size': [-1, -1], 'type': [torch.Tensor]},
                >>>                     'question_length': {'size': [-1], 'type': [list, int]},
                >>>                     # ...
                >>>                     }

                Please indicate both the size and the type as ``lists``:

                    - Indicate all dimensions in the correct order for each key `size` field. If a dimension is\
                    unimportant or unknown (e.g. the batch size or variable-length sequences), then please indicate \
                    ``-1`` at the correct location.
                    - If an object is a composition of several Python objects (``list``, ``dict``,...), then please \
                    include all objects type, matching the dimensions order: e.g. ``[list, str]``.


        :param problem_data_definitions_: Contains the definition of a sample generated by the ``Problem`` class.
        :type problem_data_definitions_: dict

        :return: True if the ``Model`` accepts what the ``Problem`` generates, otherwise throws an exception.
        """

        for key in self.data_definitions.keys():

            if key not in problem_data_definitions_.keys():
                raise KeyError('The key {} is missing in the Problem.data_definitions. Handshake failed.'.format(key))

            else:
                # key exists, first check the size:
                for i, dim in enumerate(self.data_definitions[key]['size']):
                    if dim == -1:
                        pass  # don't care
                    else:  # we actually need to check
                        if dim != problem_data_definitions_[key]['size'][i]:
                            raise ValueError('There is a mismatch in the expected size of the key {} '
                                             'in Problem.data_definitions. Expected {}, got {}. Handshake failed.'.format(
                                key, dim, problem_data_definitions_[key]['size'][i]))

                # then, check the type:
                for i, tp in enumerate(self.data_definitions[key]['type']):
                    if not tp == problem_data_definitions_[key]['type'][i]:
                        raise ValueError('There is a mismatch in the expected type(s) of the key {} '
                                         'in Problem.data_definitions. Expected {}, got {}. Handshake failed.'.format(
                            key, tp, problem_data_definitions_[key]['type'][i]))

        # Everything matches, return true
        return True

    def add_statistics(self, stat_col):
        """
        Add statistics to collector.

        EMPTY - To be redefined in inheriting classes.

        :param stat_col: Statistics collector.

        """
        pass

    def collect_statistics(self, stat_col, data_tuple, logits):
        """
        Base statistics collection.

        EMPTY - To be redefined in inheriting classes.

        :param stat_col: Statistics collector.
        :param data_tuple: Data tuple containing inputs and targets.
        :param logits: Logits being output of the model.

        """
        pass

    @abstractmethod
    def plot(self, data_tuple, predictions, sample_number=0):
        """
        Plots inputs, targets and predictions, along with model-dependent
        variables.

        Abstract - to be defined in derived classes.

        :param data_tuple: Data tuple containing input and target batches.
        :param predictions: Prediction.
        :param sample_number: Number of sample in batch (DEFAULT: 0)

        """

    def save(self, model_dir, stat_col):
        """
        Generic method saving the model parameters to file. It can be
        overloaded if one needs more control.

        :param model_dir: Directory where the model will be saved.
        :param stat_col: Statistics collector that contain current loss and episode number (and other statistics).
        :return: True if this is the best model that is found till now (considering loss).

        """
        # Get two elementary statistics.
        loss = stat_col['loss']
        episode = stat_col['episode']

        # Checkpoint to be saved.
        chkpt = {
            'name': self.name,
            'state_dict': self.state_dict(),
            'stats': stat_col.statistics
        }

        # for key, value in stat_col.statistics.items():
        #    self.logger.warning("{}: {}".format(key, value))

        # Save the intermediate checkpoint.
        if self.save_intermediate:
            filename = model_dir + 'model_episode_{:05d}.pt'.format(episode)
            torch.save(chkpt, filename)
            self.logger.info(
                "Model and statistics exported to checkpoint {}".format(
                    filename))

        # Save the best model.
        if (loss < self.best_loss):
            self.best_loss = loss
            filename = model_dir + 'model_best.pt'
            torch.save(chkpt, filename)
            self.logger.info(
                "Model and statistics exported to checkpoint {}".format(
                    filename))
            return True
        # Else: that was not the best model.
        return False

    def load(self, checkpoint_file):
        """
        Loads model from the checkpoint file.

        :param checkpoint_file: File containing dictionary with model state and statistics.

        """
        # Load checkpoint
        # This is to be able to load CUDA-trained model on CPU
        chkpt = torch.load(
            checkpoint_file, map_location=lambda storage, loc: storage)

        # Load model.
        self.load_state_dict(chkpt['state_dict'])

        # Print statistics.
        self.logger.info(
            "Imported {} parameters from checkpoint (episode {}, loss {})".format(
                chkpt['name'],
                chkpt['stats']['episode'],
                chkpt['stats']['loss']))


if __name__ == '__main__':
    """Unit test for the handshake."""

    from utils.param_interface import ParamInterface
    params = ParamInterface()

    model = Model(params)

    # you can play with one of the dicts below to see the handshake in action.
    model.data_definitions = {'img': {'size': [-1, 320, 480, 3], 'type': [np.ndarray]},
                                 'question': {'size': [-1, -1], 'type': [torch.Tensor]},
                                 'question_length': {'size': [-1], 'type': [list, int]},
                                 'question_string': {'size': [-1,-1], 'type': [list, str]},
                                 'question_type': {'size': [-1,-1], 'type': [list, str]},
                                 'targets': {'size': [-1], 'type': [torch.Tensor]},
                                 'targets_string': {'size': [-1,-1], 'type': [list, str]},
                                 'index': {'size': [-1], 'type': [list, int]},
                                 'imgfile': {'size': [-1,-1], 'type': [list,str]},
                                 }

    problem_data_definitions = {'img': {'size': [-1, 320, 480, 3], 'type': [np.ndarray]},
                                 'question': {'size': [-1, -1], 'type': [torch.Tensor]},
                                 'question_length': {'size': [-1], 'type': [list, int]},
                                 'question_string': {'size': [-1,-1], 'type': [list, str]},
                                 'question_type': {'size': [-1,-1], 'type': [list, str]},
                                 'targets': {'size': [-1], 'type': [torch.Tensor]},
                                 'targets_string': {'size': [-1,-1], 'type': [list, str]},
                                 'index': {'size': [-1], 'type': [list, int]},
                                 'imgfile': {'size': [-1,-1], 'type': [list,str]}
                                 }

    model.handshake_definitions(problem_data_definitions)
