"""
Module to train posterior inference models using the pyDELFI package
"""

import yaml
import json
import time
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from ili.utils import load_class, load_from_config
from .pydelfi_wrappers import DelfiWrapper
from .base import _BaseRunner


class DelfiRunner(_BaseRunner):
    """Class to train posterior inference models using the pydelfi package

    Args:
        prior (Any): prior on the parameters
        engine_kwargs (Dict): dictionary of additional keywords for Delfi
            engine
        train_args (Dict): dictionary of hyperparameters for training
        out_dir (str, Path): directory where to store outputs
    """

    def __init__(
        self,
        prior: Any,
        config_ndes: List[Dict],
        engine_kwargs: Dict = {},
        train_args: Dict = {},
        out_dir: Union[str, Path] = None,
        device: str = 'cpu',
        name: Optional[str] = "",
    ):
        super().__init__(
            prior=prior,
            train_args=train_args,
            out_dir=out_dir,
            device=device,
            name=name,
        )
        self.config_ndes = config_ndes
        self.engine_kwargs = engine_kwargs
        self.inference_class = DelfiWrapper
        self.engine = 'NLE'
        if device != 'cpu':
            logging.warning(
                'pydelfi only supports cpu training. Device set to cpu.')
            self.device = 'cpu'

    @classmethod
    def from_config(cls, config_path: Path, **kwargs) -> "DelfiRunner":
        """Create an sbi runner from a yaml config file

        Args:
            config_path (Path, optional): path to config file.
            **kwargs: optional keyword arguments to overload config file
        Returns:
            DelfiRunner: the pyDELFI runner specified by the config file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)

        # optionally overload config with kwargs
        config.update(kwargs)

        # currently, all arguments of pyDELFI priors must be np.arrays
        for k, v in config["prior"]["args"].items():
            config["prior"]["args"][k] = np.array(v)
        prior = load_from_config(config["prior"])

        config_ndes = config["model"]["nets"]
        if 'kwargs' in config["model"]:
            engine_kwargs = config["model"]["kwargs"]
        else:
            engine_kwargs = {}

        # load logistics
        train_args = config["train_args"]
        out_dir = Path(config["out_dir"])
        if "name" in config["model"]:
            name = config["model"]["name"] + "_"
        else:
            name = ""
        signatures = []
        for type_nn in config_ndes:
            signatures.append(type_nn.pop("signature", ""))
        return cls(
            prior=prior,
            config_ndes=config_ndes,
            engine_kwargs=engine_kwargs,
            train_args=train_args,
            out_dir=out_dir,
            name=name,
        )

    def _save_models(self, posterior: DelfiWrapper, summary: Dict[str, Any]):
        """Save the trained models to file"""
        logging.info(f"Saving models to {self.out_dir}")
        str_p = self.name + "posterior.pkl"
        str_s = self.name + "summary.json"
        posterior.save_engine(str_p)
        with open(self.out_dir / str_s, "w") as f:
            json.dump(summary, f)

    def __call__(self, loader):
        """Train your posterior and save it to file

        Args:
            loader (BaseLoader): dataloader with stored data-parameter pairs
        """

        t0 = time.time()
        x = loader.get_all_data()
        theta = loader.get_all_parameters()

        n_params = theta.shape[-1]
        n_data = x.shape[-1]

        nets = self.inference_class.load_ndes(
            n_params=n_params,
            n_data=n_data,
            config_ndes=self.config_ndes,
        )

        posterior = self.inference_class(
            config_ndes=self.config_ndes,
            data=x[0],
            prior=self.prior,
            nde=nets,
            name=self.name,
            results_dir=str(self.out_dir)+'/',
            param_names=np.arange(n_params).astype(str),
            graph_restore_filename="graph_checkpoint",
            restore_filename="temp.pkl",
            restore=False, save=True,
            **self.engine_kwargs,
        )
        posterior.load_simulations(x, theta)
        posterior.train_ndes(**self.train_args)

        train_probs = [(-t).tolist() for t in posterior.training_loss]
        val_probs = [(-t).tolist() for t in posterior.validation_loss]
        summary = dict(
            training_log_probs=train_probs,
            validation_log_probs=val_probs,
            epochs_trained=[len(posterior.training_loss[0])]
        )

        if self.out_dir is not None:
            self._save_models(posterior, summary)
        tf.reset_default_graph()

        logging.info(
            f"It took {time.time() - t0} seconds to train all models.")

        return posterior, summary
