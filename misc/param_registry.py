from misc.singleton import SingletonMetaClass
from collections import Mapping
from abc import ABCMeta
import yaml


class MetaSingletonABC(SingletonMetaClass, ABCMeta):
    """
    Metaclass that inherits both SingletonMetaClass, and ABCMeta
    (collection.Mappings' metaclass).
    """
    pass


class ParamRegistry(Mapping, metaclass=MetaSingletonABC):
    """
    This class should not be used except through `ParameterInterface`.

    Registry singleton for the parameters. Registers default values (from the models, problems, etc) as well as custom
    values loaded by the user for the particular experiment.

    Parameters can be read from the registry by indexing. The returned parameters are the default ones superseded by
    all the custom ones. The merging of default and custom parameters is computed every time the parameters are read.

    """

    def __init__(self):
        super(ParamRegistry, self).__init__()
        self._default_params = {}
        self._superseding_params = {}
        self._params = dict()

    def _update_params(self):
        """
        Computes the params from the default param registry superseded by the
        custom params registry.

        :return: None

        """
        self._params = self._default_params.copy()
        self.update_dict_recursively(self._params, self._superseding_params)

    def add_default_params(self, default_params: dict):
        """
        Appends default params to the registry. This should not be used by the
        user, but rather set by the objects necessitating default values.

        :param default_params: Dictionary containing default values.
        :return: None

        """
        self.update_dict_recursively(self._default_params, default_params)
        self._update_params()

    def add_custom_params(self, custom_params: dict):
        """
        Appends custom parameters to the registry. This is intended for the
        user to customize the experiments.

        :param custom_params: Dictionary containing custom values.
        :return: None

        """
        self.update_dict_recursively(self._superseding_params, custom_params)
        self._update_params()

    def __getitem__(self, key):
        """
        Get parameter value under key. The parameter dict is derived from the
        default parameters updated with the custom parameters.

        :param key: key to value in parameters
        :return: parameter value

        """
        return self._params[key]

    def __iter__(self):
        return iter(self._params)

    def __len__(self):
        return len(self._params)

    def add_custom_params_from_yaml(self, yaml_path: str):
        """
        Helper function. Has the same functionality as `add_custom_params`, but
        loads from path of yaml file.

        :param yaml_path: Path to yaml file containing custom paramters.
        :return: None

        """
        with open(yaml_path, 'r') as stream:
            params_from_yaml = yaml.load(stream)

        self.add_custom_params(params_from_yaml)

    def update_dict_recursively(self, d, u):
        for k, v in u.items():
            if isinstance(v, Mapping):
                d[k] = self.update_dict_recursively(d.get(k, {}), v)
            else:
                d[k] = v
        return d
