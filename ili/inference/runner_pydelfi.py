"""
Module to train posterior inference models using the pyDELFI package
"""

import yaml
import time
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
from ili.utils import load_class, load_from_config


class DelfiRunner():
    """Class to train posterior inference models using the pydelfi package

    Args:
        prior (Independent): prior on the parameters
        inference_class (Any): pydelfi inference class used to that train
            neural posteriors
        engine_kwargs (Dict): dictionary of additional keywords for Delfi
            engine
        train_args (Dict): dictionary of hyperparameters for training
        out_dir (Path): directory where to store outputs
    """

    def __init__(
        self,
        config_ndes: List[Dict],
        prior: Any,
        inference_class: Any,
        out_dir: Path,
        engine_kwargs: Dict = {},
        train_args: Dict = {},
        name: Optional[str] = ""
    ):
        self.config_ndes = config_ndes
        self.prior = prior
        self.inference_class = inference_class
        self.engine_kwargs = engine_kwargs
        self.train_args = train_args
        self.out_dir = out_dir
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.name = name

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
        inference_class = load_class(
            module_name=config["model"]["module"],
            class_name=config["model"]["class"],
        )

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
            config_ndes=config_ndes,
            prior=prior,
            inference_class=inference_class,
            engine_kwargs=engine_kwargs,
            train_args=train_args,
            out_dir=out_dir,
            name=name,
        )

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

        posterior.save_engine(self.name+'posterior.pkl')
        tf.reset_default_graph()

        logging.info(
            f"It took {time.time() - t0} seconds to train all models.")
