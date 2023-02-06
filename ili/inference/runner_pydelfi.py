import yaml
import time
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Callable
from ili.utils import load_class, load_from_config


class DelfiRunner:
    def __init__(
        self,
        n_params: int,
        n_data: int,
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
            embedding_net (nn.Module): neural network to compress high dimensional data into lower dimensionality
            train_args (Dict): dictionary of hyperparameters for training
            output_path (Path): path where to store outputs
        """
        self.n_params = n_params
        self.n_data = n_data
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
        
        n_params = config['n_params']
        n_data = config['n_data']
        
        # currently, all arguments of pyDELFI must be np.arrays
        for k, v in config["prior"]["args"].items():
            config["prior"]["args"][k] = np.array(v)
        prior = load_from_config(config["prior"])
        
        inference_class = load_class(
            module_name=config["model"]["module"],
            class_name=config["model"]["class"],
        )
        neural_posteriors = cls.load_neural_posteriors(
            posteriors_config=config["model"]["neural_posteriors"],
            n_params=n_params,
            n_data=n_data
        )
        train_args = config["train_args"]
        output_path = Path(config["output_path"])
        return cls(
            n_params=n_params,
            n_data=n_data,
            prior=prior,
            inference_class=inference_class,
            neural_posteriors=neural_posteriors,
            train_args=train_args,
            output_path=output_path,
        )


    @classmethod
    def load_neural_posteriors(
        cls,
        posteriors_config: List[Dict],
        n_params: int,
        n_data: int,
    ) -> List[Callable]:
        """Load the inference model

        Args:
            posterior_config(List[Dict]): list with configurations for each neural posterior
            model in the ensemble

        Returns:
            List[Callable]: list of neural posterior models with forward methods
        """
        neural_posteriors = []
        for i, model_args in enumerate(posteriors_config):
            model_args['args']['index'] = i
            model_args['args']['n_parameters'] = n_params
            model_args['args']['n_data'] = n_data
            # layer activations must be input as TF classes
            if 'act_fun' in model_args['args']:
                model_args['args']['act_fun'] = load_class('tensorflow', model_args['args']['act_fun'])
            elif 'activations' in model_args['args']:
                model_args['args']['activations'] = [load_class('tensorflow', x) for x in model_args['args']['activations']]
            
            neural_posteriors.append(
                load_from_config(model_args)
            )
        return neural_posteriors

    
    def __call__(self, loader):
        """Train your posterior and save it to file

        Args:
            loader (BaseLoader): data loader with stored summary-parameter pairs
        """

        t0 = time.time()
        x = loader.get_all_data()
        theta = loader.get_all_parameters()
        
        posterior = self.inference_class(
            x[0], self.prior, self.neural_posteriors,
            results_dir = str(self.output_path),
            param_names = np.arange(self.n_params).astype(str)
        )
        posterior.load_simulations(x, theta)
        posterior.train_ndes(**self.train_args)
        
        with open(self.output_path / "posterior.pkl", "wb") as handle:
            pickle.dump(posterior, handle)
        logging.info(f"It took {time.time() - t0} seconds to train all models.")
