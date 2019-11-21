import importlib
import json
import os
from builtins import object

from ConfigSpace.conditions import InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, \
    CategoricalHyperparameter
from typing import Dict, Any

from sklearn.base import BaseEstimator


class Algorithm(object):
    def __init__(self, method):
        """
        method: method code or path to JSON file containing all the information
            needed to specify this enumerator.
        """
        config_path = os.path.join(os.path.dirname(__file__), 'methods', method)
        with open(config_path) as f:
            config = json.load(f)

        self.root_params = config['root_hyperparameters']
        self.conditions = config['conditional_hyperparameters']
        self.class_path = config['class']

        # create hyperparameters from the parameter config
        cs = ConfigurationSpace()

        self.parameters = {}
        for k, v in list(config['hyperparameters'].items()):
            param_type = v['type']
            if param_type == 'int':
                self.parameters[k] = UniformIntegerHyperparameter(k, lower=v['range'][0], upper=v['range'][1],
                                                                  default_value=v.get('default', None))
            elif param_type == 'int_exp':
                self.parameters[k] = UniformIntegerHyperparameter(k, lower=v['range'][0], upper=v['range'][1],
                                                                  default_value=v.get('default', None), log=True)
            elif param_type == 'float':
                self.parameters[k] = UniformFloatHyperparameter(k, lower=v['range'][0], upper=v['range'][1],
                                                                default_value=v.get('default', None))
            elif param_type == 'float_exp':
                self.parameters[k] = UniformFloatHyperparameter(k, lower=v['range'][0], upper=v['range'][1],
                                                                default_value=v.get('default', None), log=True)
            elif param_type == 'string':
                # noinspection PyArgumentList
                self.parameters[k] = CategoricalHyperparameter(k, choices=v['values'],
                                                               default_value=v.get('default', None))
            elif param_type == 'bool':
                # noinspection PyArgumentList
                self.parameters[k] = CategoricalHyperparameter(k, choices=[True, False],
                                                               default_value=v.get('default', None))
            else:
                raise ValueError(f'Unknown hyperparameter type {param_type}')

        cs.add_hyperparameters(self.parameters.values())

        for condition, dic in self.conditions.items():
            for k, v in dic.items():
                cs.add_condition(InCondition(self.parameters[k], self.parameters[condition], v))

        self.cs = cs

    def random_config(self):
        return self.cs.sample_configuration()

    def default_config(self):
        return self.cs.get_default_configuration()

    def instance(self, params: Dict[str, Any] = None) -> BaseEstimator:
        if params is None:
            params = self.random_config()

        module_name = self.class_path.rpartition(".")[0]
        class_name = self.class_path.split(".")[-1]

        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        return class_(**params)


ALGORITHMS = {}

if len(ALGORITHMS) == 0:
    directory = os.path.join(os.path.dirname(__file__), 'methods')
    for file_name in os.listdir(directory):
        a = Algorithm(file_name)
        ALGORITHMS[a.class_path] = a
