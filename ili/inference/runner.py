
"""
Module to contain a universal inference engine configuration for all backends.
"""
import yaml
from typing import Any
from pathlib import Path

try:
    from ili.inference import SBIRunner, SBIRunnerSequential, ABCRunner
    backend_ml = 'torch'
except ImportError:
    from ili.inference import DelfiRunner
    backend_ml = 'tensorflow'


class InferenceRunner():
    """ A universal class to train posterior inference models using either
        the sbi or pydelfi backends. Provides a univeral interface to configure
        either backend.
    """

    def __init__(
        self,
        backend: str,
        engine: str,
        prior: Any,
        out_dir: Path = None,
        device: str = 'cpu',
        name: str = '',
        **kwargs
    ):
        """Initialize the InferenceRunner

        Args:
            config_path (Any): path to config file
            backend (str, optional): backend to use. Defaults to 'torch'.
            **kwargs: optional keyword arguments to overload config file
        """
        engine_class = self._parse_engine(backend, engine)

        return engine_class(
            prior=prior,
            out_dir=out_dir,
            device=device,
            name=name,
            **kwargs
        )

    @classmethod
    def from_config(cls, config_path: Path, **kwargs) -> "InferenceRunner":
        """Create an sbi runner from a yaml config file

        Args:
            config_path (Path, optional): path to config file.
            **kwargs: optional keyword arguments to overload config file
        Returns:
            InferenceRunner: the inference runner specified by the config file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)

        # optionally overload config file with kwargs
        config.update(kwargs)

        backend = config['model']['backend']
        engine = config['model']['engine']

        inference_class = cls._parse_engine(backend, engine)

        if backend == 'sbi':
            config['model']['module'] = 'sbi.inference'
            config['model']['class'] = (engine if engine[0] == 'S'
                                        else f"S{engine}")
        elif backend == 'pydelfi':
            config['model']['module'] = 'ili.inference.pydelfi_wrappers'
            config['model']['class'] = 'DelfiWrapper'

        return inference_class.from_config(config_path, **config)

    @staticmethod
    def _parse_engine(backend: str, engine: str) -> Any:
        """Parse the backend and engine to load the correct class

        Args:
            backend (str): name of the backend (sbi or pydelfi)
            engine (str): name of the engine class (NPE/NLE/NRE or SNPE/SNLE/SNRE)

        Returns:
            Any: the loaded engine class
        """
        global backend_ml

        if backend == 'sbi':
            if backend_ml != 'torch':  # check installation
                raise ValueError(
                    'User requested an sbi model, but torch backend is not '
                    'installed. Please use torch installation or change model.'
                )
            # check model type
            if engine not in ['NPE', 'NLE', 'NRE', 'SNPE', 'SNLE', 'SNRE']:
                raise ValueError(
                    'User requested an invalid model type for sbi: '
                    f'{engine}. Please use one of: NPE, NLE, NRE,  '
                    'SNPE, SNLE, or SNRE.'
                )

            if engine[0] == 'S':
                return SBIRunnerSequential
            else:
                return SBIRunner
        elif backend == 'pydelfi':
            if backend_ml != 'tensorflow':  # check installation
                raise ValueError(
                    'User requested a pydelfi model, but tensorflow is not '
                    'installed. Please use tensorflow installation or change '
                    'model.'
                )
            # check model type
            if engine not in ['NLE', 'SNLE']:
                raise ValueError(
                    'User requested an invalid model type for pydelfi: '
                    f'{engine}. Please use either NLE or SNLE.'
                )

            return DelfiRunner
        else:
            raise ValueError(
                f'User requested an invalid model backend: {backend}. Please '
                'use either sbi or pydelfi.'
            )
