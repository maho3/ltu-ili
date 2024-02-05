
"""
Module to contain a universal inference engine configuration for all backends.
"""
import yaml
from typing import Any, Union
from pathlib import Path
from ili.utils import update

try:
    from ili.inference import SBIRunner, SBIRunnerSequential, LampeRunner
    interface = 'torch'
except ImportError:
    from ili.inference import DelfiRunner
    interface = 'tensorflow'


class InferenceRunner():
    """ A universal class to train posterior inference models using either
        the sbi/pydelfi/lampe backends. Provides a univeral interface to configure
        either backend.
    """

    def __init__(self):
        raise NotImplementedError(
            'This class should not be instantiated. Did you mean to use '
            '.load() or .from_config()?'
        )

    @classmethod
    def load(
        cls,
        backend: str,
        engine: str,
        prior: Any,
        out_dir: Union[str, Path] = None,
        device: str = 'cpu',
        name: str = '',
        **kwargs
    ):
        """Create an inference runner from inline arguments

        Args:
            backend (str): name of the backend (sbi/pydelfi/lampe)
            engine (str): name of the engine class (NPE/NLE/NRE or SNPE/SNLE/SNRE)
            prior (Any): prior distribution
            out_dir (str, Path, optional): path to output directory. Defaults to None.
            device (str, optional): device to run on. Defaults to 'cpu'.
            name (str, optional): name of the runner. Defaults to ''.
            **kwargs: optional keyword arguments to specify to the runners
        """
        runner_class = cls._parse_runner(backend, engine)

        return runner_class(
            prior=prior,
            out_dir=out_dir,
            device=device,
            name=name,
            engine=engine,
            **kwargs
        )

    @classmethod
    def from_config(cls, config_path: Path, **kwargs) -> "InferenceRunner":
        """Create an inference runner from a yaml config file

        Args:
            config_path (Path, optional): path to config file.
            **kwargs: optional keyword arguments to overload config file
        Returns:
            InferenceRunner: the inference runner specified by the config file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)

        # optionally overload config file with kwargs
        update(config, **kwargs)

        backend = config['model']['backend']
        engine = config['model']['engine']

        runner_class = cls._parse_runner(backend, engine)

        return runner_class.from_config(config_path, **config)

    @staticmethod
    def _parse_runner(backend: str, engine: str) -> Any:
        """Parse the backend and engine to load the correct class

        Args:
            backend (str): name of the backend (sbi or pydelfi)
            engine (str): name of the engine class (NPE/NLE/NRE or SNPE/SNLE/SNRE)

        Returns:
            Any: the loaded training class
            Any: the loaded engine class
        """
        global interface

        if backend == 'sbi':
            if interface != 'torch':  # check installation
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
            if interface != 'tensorflow':  # check installation
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
        elif backend == 'lampe':
            if interface != 'torch':  # check installation
                raise ValueError(
                    'User requested a lampe model, but torch backend is not '
                    'installed. Please use torch installation or change model.'
                )
            # check model type
            if engine not in ['NPE']:
                raise ValueError(
                    'User requested an invalid model type for lampe: '
                    f'{engine}. lampe only supports NPE.'
                )
            return LampeRunner
        else:
            raise ValueError(
                f'User requested an invalid model backend: {backend}. Please '
                'use either sbi or pydelfi.'
            )
