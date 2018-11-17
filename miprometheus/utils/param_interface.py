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

import yaml
from collections import Mapping
from miprometheus.utils.param_registry import ParamRegistry


class ParamInterface(Mapping):
    """
    Interface to the :py:class:`ParamRegistry` singleton.

    Inherits :py:class:`collections.Mapping`, and therefore exposes functionality close to a `dict`.

    Offers a read (through :py:class:`collections.Mapping` interface) and write \
    (through :py:func:`add_default_params` and :py:func:`add_config_params` methods) \
    view of the :py:class:`ParamRegistry`.

    """

    def __init__(self, *keys):
        """
        Constructor:
            - Call base constructor (:py:class:`Mapping`),
            - Initializes the :py:class:`ParamRegistry`,
            - Initializes empty keys_path list
            - **Adds the recursive dict structure determined by the given keys to the default params**


        :param keys: Sequence of keys to the subtree of the registry. The subtree hierarchy will be created if it \
        does not exist. If empty, shows the whole registry.
        :type keys: sequence / collection: dict, list etc.

        """
        # call base constructor
        super(ParamInterface, self).__init__()

        # empty ParamRegistry
        self._param_registry = ParamRegistry()

        # keys_path as a list
        self._keys_path = list(keys)

clear
gdef _lookup(self, *keys):
        """
        Returns the :py:class:`ParamInterface` living under ``keys``.

        :param keys: Sequence of keys to the subtree of the registry. If empty, shows the whole registry.
        :type keys: sequence / collection: dict, list etc.

        """

        def lookup_recursion(dic, key, *keys):
            if keys:
                return lookup_recursion(dic[key], *keys)
            return dic[key]

        # construct the path from the existing keys path
        lookup_keys = self._keys_path + list(keys)

        if len(lookup_keys) > 0:
            r = lookup_recursion(self._param_registry, *lookup_keys)
            return r
        else:
            return self._param_registry

    def _nest_dict(self, d: dict):
        """
        Create a nested dict using a sequence of
        :param d:
        :return:
        """

        def nest_dict_recursion(dic, key, *keys):
            if keys:
                dic[key] = {}
                return nest_dict_recursion(dic[key], *keys)
            else:
                dic[key] = {}
                dic[key].update(d)

        if len(self._keys_path) > 0:
            nested_dict = {}
            nest_dict_recursion(nested_dict, *self._keys_path)
            return nested_dict
        else:
            return d

    def to_dict(self):
        """

        :return: `dict` containing a snapshot of the current :py:class:`ParamInterface` tree.
        """
        return dict(self._lookup())

    def __getitem__(self, key):
        """
        Get parameter value under key. The parameter dict is derived from the \
        default parameters updated with the config parameters.

        :param key: key to value in parameters
        :return: ``ParameterInterface(key)`` or value if leaf of the ``ParamRegistry`` tree.

        """
        v = self._lookup(key)
        if isinstance(v, dict) or isinstance(v, ParamRegistry):
            return ParamInterface(*self._keys_path, key)
        else:  # We are at a leaf of the tree
            return v

    def __len__(self):
        return len(self._lookup())

    def __iter__(self):
        return iter(self._lookup())

    def add_default_params(self, default_params: dict):
        """
        Appends default params dictionary to the registry. This should not be \
        used by the user, but rather set by the objects necessitating default \
        values. The dictionary will be inserted into the subtree chosen during \
        initialization of ``ParameterInterface``

        :param default_params: Dictionary containing default values.

        """
        self._param_registry.add_default_params(
            self._nest_dict(default_params)
        )

    def add_config_params(self, config_params: dict):
        """
        Appends config parameters dictionary to the registry. This is intended \
        for the user to dynamically (re)configure the experiments. The dictionary \
        will be inserted into the subtree chosen during initialization of \
        ``ParameterInterface``

        :param config_params: Dictionary containing config values.


        """
        self._param_registry.add_config_params(
            self._nest_dict(config_params)
        )

    def del_default_params(self, key):
        """
        Removes an entry from the Default Parameters. \
        The entry can either be a subtree or a leaf of the Default Parameters tree.

        :param key: key to subtree / leaf in the Default Parameters

        """
        self._param_registry.del_default_params(self._keys_path + [key])

    def del_config_params(self, key):
        """
        Removes an entry from the Configuration Parameters. \
        The entry can either be a subtree or a leaf of the Configuration Parameters tree.

        :param key: key to subtree / leaf in the Configuration Parameters

        """
        self._param_registry.del_config_params(self._keys_path + [key])

    def add_config_params_from_yaml(self, yaml_path: str):
        """
        Helper function. Has the same functionality as ``add_config_params``, but \
        loads from path of yaml file.

        :param yaml_path: Path to yaml file containing config parameters.

        """
        # Open file and try to add that to list of parameter dictionaries.
        with open(yaml_path, 'r') as stream:
            # Load parameters.
            params_from_yaml = yaml.load(stream)

        self.add_config_params(params_from_yaml)


if __name__ == '__main__':
    # Test code
    pi0 = ParamInterface()
    pi1 = ParamInterface('level0', 'level1')

    pi0.add_default_params({
        'param0': "0_from_code",
        'param1': "1_from_code"
    })

    print('pi0', pi0.to_dict())

    pi0.add_config_params({
        'param1': "-1_from_config_file"
    })

    print('pi0', pi0.to_dict())

    pi1.add_default_params({
        'param2': 2,
        'param3': 3
    })

    print('pi0', pi0.to_dict())
    print('pi1', pi1.to_dict())

    pi1.add_config_params({
        'param2': -2
    })

    print('pi0', pi0.to_dict())
    print('pi1', pi1.to_dict())

    pi2 = pi0['level0']
    print('pi2', pi2.to_dict())

    pi1.add_config_params({
        'param2': -3
    })

    print('pi2', pi2.to_dict())

    pi3 = pi0['level0']['level1']

    print('pi3', pi3.to_dict())
