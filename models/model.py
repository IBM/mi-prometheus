#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""model.py: contains base abstract model for all models"""
__author__ = "Tomasz Kornuta, Vincent Marois"

import torch
from torch import nn
from abc import abstractmethod
import numpy as np

import logging

from utils.app_state import AppState


class Model(nn.Module):
    """
    Class representing base class for all Models.

    Inherits from torch.nn.Module as all subclasses will represent a trainable model.

    Hence, all subclasses should override the ``forward`` function.

    Implements features & attributes used by all subclasses.

    """

    def __init__(self, params, problem_default_values_={}):
        """
        Initializes a Model object.

        :param params: Parameters read from configuration file.

        :param problem_default_values_: dict of parameters values coming from the problem class. One example of such\
        parameter value is the size of the vocabulary set in a translation problem.
        :type problem_default_values_: dict

        This constructor:

        - stores a pointer to ``params``:

            >>> self.params = params

        - sets a default problem name:

            >>> self.name = 'Model'

        - initializes the logger.

            >>> self.logger = logging.Logger(self.name)

        - tries to parse the values coming from ``problem_default_values_``:

            >>>         try:
            >>>             for key in problem_default_values_.keys():
            >>>                 self.params.add_custom_params({key: problem_default_values_[key]})
            >>>         except BaseException:
            >>>             self.logger.info('No parameter value was parsed from problem_default_values_')

        - initializes the data definitions:

        .. note::

            This dict contains information about the expected inputs and produced outputs of the current model class.

            This object will be used during handshaking between the model and the problem class to ensure that the model
            can accept the batches produced by the problem and that the problem can accept the predictions of the model
            to compute the loss and accuracy.

            This dict should be defined using self.params.

            This dict should at least contains the `targets` field:

                >>>     self.data_definitions = {'inputs': {'size': [-1, -1], 'type': [torch.Tensor]},
                >>>                              'targets': {'size': [-1, 1], 'type': [torch.Tensor]}
                >>>                             }


        - sets the access to ``AppState``: for dtype, visualization flag etc.

            >>> self.app_state = AppState()

        - initialize the best model loss (to select which model to save) to ``np.inf``:

            >>> self.best_loss = np.inf

        """
        # Call base class constructor here.
        super(Model, self).__init__()

        # Store pointer to params.
        self.params = params

        # "Default" model name.
        self.name = 'Model'

        # initialize the logger
        self.logger = logging.getLogger(self.name)

        # Flag indicating whether intermediate checkpoints should be saved or
        # not (DEFAULT: False).
        if "save_intermediate" not in params:
            params.add_default_params({"save_intermediate": False})
        self.save_intermediate = params["save_intermediate"]

        # process all params from configuration file and problem_default_values_ here
        try:
            for key in problem_default_values_.keys():
                self.params.add_custom_params({key: problem_default_values_[key]})

        except BaseException:
            self.logger.info('No parameter value was parsed from problem_default_values_')

        # --> We assume from here that the model class has all parameters values needed (either from params or
        # problem_default_values_) to be correctly instantiated and contained in self.params.

        # We can then define a dict that contains a description of the expected (and mandatory) inputs for this model.
        # This dict should be defined using self.params.
        self.data_definitions = {'inputs': {'size': [-1, -1], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1, 1], 'type': [torch.Tensor]}
                                }

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

                - Verifying that all keys existing in ``Model.data_definitions`` are also existing in \
                  ``Problem.data_definitions``. If a key is missing, an exception is thrown.

                  This function does not verify the key ``targets`` as this will be done by\
                   ``problems.problem.Problem.handshake_definitions``.

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
                    include all objects type, matching the dimensions order: e.g. ``[list, dict]``.


        :param problem_data_definitions_: Contains the definition of a sample generated by the ``Problem`` class.
        :type problem_data_definitions_: dict

        :return: True if the ``Model`` accepts what the ``Problem`` generates, otherwise throws an exception.
        """

        for key in [k for k in self.data_definitions.keys() if k != 'targets']:

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
        Adds statistics to ``StatisticsCollector``.

        .. note::


            Empty - To be redefined in inheriting classes.


        :param stat_col: ``StatisticsCollector``.

        """
        pass

    def collect_statistics(self, stat_col, data_dict, logits):
        """
        Base statistics collection.

         .. note::


            Empty - To be redefined in inheriting classes. The user has to ensure that the corresponding entry \
            in the ``StatisticsCollector`` has been created with ``self.add_statistics()`` beforehand.

        :param stat_col: ``StatisticsCollector``.

        :param data_dict: ``DataDict`` containing inputs and targets.
        :type data_dict: DataDict

        :param logits: Predictions being output of the model.

        """
        pass

    @abstractmethod
    def plot(self, data_dict, predictions, sample=0):
        """
        Plots inputs, targets and predictions, along with model-dependent
        variables.

        . note::

             Abstract - to be defined in derived classes.

        :param data_dict: DataDict containing input and target batches.
        :type data_dict: DataDict

        :param predictions: Prediction.
        :type predictions: torch.tensor

        :param sample: Number of sample in batch (default: 0)
        :type sample: int

        """

    def save(self, model_dir, stat_col):
        """
        Generic method saving the model parameters to file. It can be
        overloaded if one needs more control.

        :param model_dir: Directory where the model will be saved.
        :type model_dir: str

        :param stat_col: Statistics collector containing the current loss and episode number (among other statistics).
        :type stat_col: StatisticsCollector

        :return: True if this is currently the best model (until the current episode, considering the loss).

        """
        # Get the two elementary statistics.
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
        if loss < self.best_loss:
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
        Loads a model from the specified checkpoint file.

        :param checkpoint_file: File containing dictionary with model state and statistics.

        """
        # Load checkpoint
        # This is to be able to load a CUDA-trained model on CPU
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


    def summarize(self):
        """Summarizes model by showing trainable/non-trainable parameters and weights.
            Uses recursive_summarize to interate through nested structure of the mode.

            :param: Model object for which the summary will be created.
                
        """
        #add name of the current module
        summary_str = '\n' + '='*80 + '\n'
        summary_str += 'Model name (Type) \n'
        summary_str += '  + Submodule name (Type) \n'
        summary_str += '      Matrices: [(name, dims), ...]\n'
        summary_str += '      Trainable Params: #\n'
        summary_str += '      Non-trainable Params: #\n'
        summary_str += '='*80 + '\n'
        summary_str += self.recursive_summarize(self, 0, self.name)
        # Sum the model parameters.
        num_total_params = sum([np.prod(p.size()) for p in self.parameters()])
        mod_trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        num_trainable_params = sum([np.prod(p.size()) for p in mod_trainable_params])
        summary_str += '\nTotal Trainable Params: {}\n'.format(num_trainable_params)
        summary_str += 'Total Non-trainable Params: {}\n'.format(num_total_params-num_trainable_params) 
        summary_str += '='*80 + '\n'
        return summary_str

    def recursive_summarize(self, module_, indent_, module_name_):
        """
            Function that recursively inspects the (sub)modules and records their statistics (like names, types, parameters, their numbers etc.) 

            :param module_: Module to be inspected.
            :param indent_: Current indentation level.
            :param module_name_: Name of the module that will be displayed before its type. 

        """
        # Recursivelly inspect the children.
        child_lines = []
        for key, module in module_._modules.items():
            child_lines.append(self.recursive_summarize(module, indent_+1, key))

        # "Leaf information". 
        mod_str = ''
        if indent_ > 0:
            mod_str += '  ' + '| '*(indent_-1) + '+ '
        mod_str += module_name_ + " ("+ module_._get_name() + ')'
        #mod_str += '-'*(80 - len(mod_str))
        mod_str += '\n'
        mod_str += ''.join(child_lines)
        # Get leaf weights and number of params - only for leafs!
        if not child_lines:
            # Collect names and dimensions of all (named) params. 
            mod_weights = [(n,tuple(p.size())) for n,p in module_.named_parameters()]
            mod_str += '  ' + '| '* (indent_) + '  Matrices: {}\n'.format(mod_weights)
            # Sum the parameters.
            num_total_params = sum([np.prod(p.size()) for p in module_.parameters()])
            mod_trainable_params = filter(lambda p: p.requires_grad, module_.parameters())
            num_trainable_params = sum([np.prod(p.size()) for p in mod_trainable_params])
            mod_str += '  ' + '| '* (indent_) + '  Trainable Params: {}\n'.format(num_trainable_params)
            mod_str += '  ' + '| '* (indent_) + '  Non-trainable Params: {}\n'.format(num_total_params-num_trainable_params) 
            mod_str += '  ' + '| '* (indent_) + '\n'
    
        return mod_str


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
