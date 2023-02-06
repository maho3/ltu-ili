import yaml
import time
import logging
import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, List, Callable
from ili.utils import load_class, load_from_config


class DelfiRunner:
    def __init__(
        self,
        n_params: int,
        n_data: int,
        config_ndes: List[Dict],
        prior: Any,
        inference_class: Any,
        neural_posteriors: List[Callable],
        train_args: Dict,
        output_path: Path,
    ):
        """Class to train posterior inference models using the pyDELFI package

        Args:
            prior (Independent): prior on the parameters
            inference_class (Any): pydelfi inference class used to that train neural posteriors
            neural_posteriors (List[Callable]): list of neural posteriors to train
            train_args (Dict): dictionary of hyperparameters for training
            output_path (Path): path where to store outputs
        """
        self.n_params = n_params
        self.n_data = n_data
        self.config_ndes = config_ndes
        self.prior = prior
        self.inference_class = inference_class
        self.neural_posteriors = neural_posteriors
        self.train_args = train_args
        self.output_path = output_path
        if self.output_path is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config_path) -> "DelfiRunner":
        """Create an sbi runner from a yaml config file

        Args:
            config_path (Path, optional): path to config file. Defaults to default_config.
        Returns:
            DelfiRunner: the pyDELFI runner specified by the config file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)
        
        # currently, all arguments of pyDELFI priors must be np.arrays
        for k, v in config["prior"]["args"].items():
            config["prior"]["args"][k] = np.array(v)
        prior = load_from_config(config["prior"])
        inference_class = load_class(
            module_name=config["model"]["module"],
            class_name=config["model"]["class"],
        )
        
        n_params = config['n_params']
        n_data = config['n_data']
        config_ndes = config["model"]["neural_posteriors"]
        neural_posteriors = inference_class.load_ndes(
            n_params=n_params,
            n_data=n_data,
            config_ndes=config_ndes,
        )
        train_args = config["train_args"]
        output_path = Path(config["output_path"])
        return cls(
            n_params=n_params,
            n_data=n_data,
            config_ndes=config_ndes,
            prior=prior,
            inference_class=inference_class,
            neural_posteriors=neural_posteriors,
            train_args=train_args,
            output_path=output_path,
        )

    
    def __call__(self, loader):
        """Train your posterior and save it to file

        Args:
            loader (BaseLoader): data loader with stored summary-parameter pairs
        """

        t0 = time.time()
        x = loader.get_all_data()
        theta = loader.get_all_parameters()
        
        posterior = self.inference_class(
            config_ndes=self.config_ndes,
            data=x[0], 
            prior=self.prior, 
            nde=self.neural_posteriors,
            results_dir=str(self.output_path)+'/',
            param_names=np.arange(self.n_params).astype(str),
            graph_restore_filename="graph_checkpoint",
            restore_filename = "posterior.pkl",
            restore=False, save=True
        )
        posterior.load_simulations(x, theta)
        posterior.train_ndes(**self.train_args)
        
        posterior.save_engine('tempmeta.pkl')
        tf.reset_default_graph()
        
        logging.info(f"It took {time.time() - t0} seconds to train all models.")
