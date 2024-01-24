"""
Module with tools for importing classes from modules and initializing them
"""

import importlib
from typing import Dict, Any


def load_class(
        module_name: str,
        class_name: str,
) -> Any:
    """General tool to load any class from any module, without initialization.

    Args:
        module_name (str): module from which to import class
        class_name (str): class name

    Returns:
        class (Any): the class of choice
    """
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_from_config(config: Dict) -> Any:
    """General tool to load and initialize any class from any module with
    given configuration.

    Args:
        config (Dict): dictionary with the configuration for the object of the
            form: {'module': Module name, 'class': Class name,
            'args': Dictionary of initialization arguments}

    Returns:
        object (Any): the object of choice
    """
    return load_class(config['module'], config['class'])(**config["args"])


def update(config: Dict, **kwargs) -> Dict:
    """Recursively update a dictionary with another dictionary.

    Args:
        config (Dict): dictionary to be updated
        **kwargs: dictionary with updates
    """
    for k, v in kwargs.items():
        if isinstance(v, dict) and k in config:
            config[k] = update(config.get(k, {}), **v)
        else:
            config[k] = v
    return config
