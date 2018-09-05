from misc.param_registry import ParamRegistry
from collections import Mapping


class ParamInterface(Mapping):
    """
    Interface to `ParameterRegistry` singleton. Inherits `collections.Mapping`, and therefore exposes functionality close
    to a `dict`.
    Offers a read (through `collections.Mapping` interface) and write (through `add_default_params` and
    `add_custom_params` methods) view of the `ParameterRegistry`.
    """

    def __init__(self, *keys):
        """

        :param keys: sequence of keys to the subtree of the registry. The subtree hierarchy will be created if it
        does not exist. If empty, shows the whole registry.
        """
        super(ParamInterface, self).__init__()
        self._param_registry = ParamRegistry()
        self._keys_path = list(keys)

        # Add the recursive dict structure determined by the given keys to
        # default params
        self.add_default_params({})

    def _lookup(self, *keys):
        def lookup_recursion(dic, key, *keys):
            if keys:
                return lookup_recursion(dic[key], *keys)
            return dic[key]

        lookup_keys = self._keys_path + list(keys)
        if len(lookup_keys) > 0:
            r = lookup_recursion(self._param_registry, *lookup_keys)
            return r
        else:
            return self._param_registry

    def _nest_dict(self, d: dict):
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

        :return: `dict` containing a snapshot of the current parameter tree.
        """
        return dict(self._lookup())

    def __getitem__(self, key):
        """
        Get parameter value under key. The parameter dict is derived from the default parameters updated with
        the custom parameters.

        :param key: key to value in parameters
        :return: ParameterInterface(key) or value if leaf of the ParamRegistry tree.
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
        Appends default params dictionary to the registry. This should not be used by the user, but rather set by the
        objects necessitating default values.
        The dictionary will be inserted into the subtree chosen during initialization of `ParameterInterface`

        :param default_params: Dictionary containing default values.
        :return: None
        """
        self._param_registry.add_default_params(
            self._nest_dict(default_params)
        )

    def add_custom_params(self, custom_params: dict):
        """
        Appends custom parameters dictionary to the registry. This is intended for the user to customize the
        experiments.
        The dictionary will be inserted into the subtree chosen during initialization of `ParameterInterface`

        :param custom_params: Dictionary containing custom values.
        :return: None
        """
        self._param_registry.add_custom_params(
            self._nest_dict(custom_params)
        )


if __name__ == '__main__':
    # Test code
    pi0 = ParamInterface()
    pi1 = ParamInterface('level0', 'level1')

    pi0.add_default_params({
        'param0': 0,
        'param1': 1
    })

    print('pi0', pi0.to_dict())

    pi0.add_custom_params({
        'param1': -1
    })

    print('pi0', pi0.to_dict())

    pi1.add_default_params({
        'param2': 2,
        'param3': 3
    })

    print('pi0', pi0.to_dict())
    print('pi1', pi1.to_dict())

    pi1.add_custom_params({
        'param2': -2
    })

    print('pi0', pi0.to_dict())
    print('pi1', pi1.to_dict())

    pi2 = pi0['level0']
    print('pi2', pi2.to_dict())

    pi1.add_custom_params({
        'param2': -3
    })

    print('pi2', pi2.to_dict())

    pi3 = pi0['level0']['level1']

    print('pi3', pi3.to_dict())
