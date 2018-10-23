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

__author__ = "Alexis Asseman, Tomasz Kornuta"

from abc import ABCMeta
from collections import Mapping
from miprometheus.utils.singleton import SingletonMetaClass


class MetaSingletonABC(SingletonMetaClass, ABCMeta):
    """
    Metaclass that inherits both SingletonMetaClass, and ABCMeta \
    (collection.Mappings' metaclass).
    """
    pass


class ParamRegistry(Mapping, metaclass=MetaSingletonABC):
    """
    This class should not be used except through ``ParameterInterface``.

    Registry singleton for the parameters. Registers default values (from workers, models, problems, etc) \
    as well as config values loaded by the user for the particular experiment.

    Parameters can be read from the registry by indexing. The returned parameters are the default ones superseded by \
    all the config ones. The merging of default and config parameters is computed every time the registry is changed.

    """

    def __init__(self):
        """
        Constructor. Call bsae constructor and initializes empty parameters dicts.

        """
        super(ParamRegistry, self).__init__()
        # Default parameters set in the code.

        self._default_params = {}
        # Parameters read from configuration files.
        self._superseding_config_params = {}
        # Resulting parameters.
        self._params = dict()

    def _update_params(self):
        """
        Computes the params from the default param registry superseded by the \
        config params registry.


        """
        self._params = self._default_params.copy()
        self.update_dict_recursively(self._params, self._superseding_config_params)

    def add_default_params(self, default_params: dict):
        """
        Appends default params (i.e. set in the code) to the registry. \
        This method should be used by the objects necessitating default values (problems, models, workers etc.). \

        :param default_params: Dictionary containing default values.


        """
        # Update default params list.
        self.update_dict_recursively(self._default_params, default_params)
        # Merge default with config list.
        self._update_params()

    def add_config_params(self, config_params: dict):
        """
        Appends parameters read from configuration files to the registry. \
        This is intended for the user to dynamically (re)configure his experiments. \

        :param config_params: Dictionary containing config values


        """
        # Update config params list.
        self.update_dict_recursively(self._superseding_config_params, config_params)
        # Merge default with config list.
        self._update_params()

    def __getitem__(self, key):
        """
        Get parameter value under key. The parameter dict is derived from the \
        default parameters updated with the config parameters.

        :param key: key to value in parameters

        :return: parameter value

        """
        return self._params[key]

    def __iter__(self):
        return iter(self._params)

    def __len__(self):
        return len(self._params)

    def update_dict_recursively(self, d, u):
        """
        Method updates a given parameter list in a recursive manner, starting \
        from the parameter registry root.

        :param d: Current parameter registry (default or config) node

        :param u: Values to be added/updated

        :return: Updated node

        """
        for k, v in u.items():
            if isinstance(v, Mapping):
                d[k] = self.update_dict_recursively(d.get(k, {}), v)
            else:
                d[k] = v
        return d
