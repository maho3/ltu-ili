import yaml
import time
import logging
import pickle
import torch
import torch.nn as nn
import sbi
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
from torch.distributions import Independent
from sbi.inference import NeuralInference
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
from ili.utils import load_class, load_from_config

logging.basicConfig(level=logging.INFO)

default_config = (
    Path(__file__).parent.parent / "examples/configs/sample_ensemble.yaml"
)


class SBIRunner:
    def __init__(
        self,
        prior: Independent,
        inference_class: NeuralInference,
        neural_posteriors: List[Callable],
        device: str,
        embedding_net: nn.Module,
        train_args: Dict,
        output_path: Path,
    ):
        """Class to train posterior inference models using the sbi package

        Args:
            prior (Independent): prior on the parameters
            inference_class (NeuralInference): sbi inference class used to that train neural posteriors
            neural_posteriors (List[Callable]): list of neural posteriors to train
            embedding_net (nn.Module): neural network to compress high dimensional data into lower dimensionality
            train_args (Dict): dictionary of hyperparameters for training
            output_path (Path): path where to store outputs
        """
        self.prior = prior
        self.inference_class = inference_class
        self.neural_posteriors = neural_posteriors
        self.device = device
        self.embedding_net = embedding_net
        self.train_args = train_args
        if 'num_round' in train_args:
            self.num_rounds = train_args['num_round']
            self.train_args.pop('num_round') 
        else:
            self.num_rounds = 1
        self.output_path = output_path
        if self.output_path is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config_path) -> "SBIRunner":
        """Create an sbi runner from a yaml config file

        Args:
            config_path (Path, optional): path to config file. Defaults to default_config.
        Returns:
            SBIRunner: the sbi runner specified by the config file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)
        prior = load_from_config(config["prior"])
        if "embedding_net" in config:
            embedding_net = load_from_config(
                config=config["embedding_net"],
            )
        else:
            embedding_net = nn.Identity()
        inference_class = load_class(
            module_name=config["model"]["module"],
            class_name=config["model"]["class"],
        )
        neural_posteriors = cls.load_neural_posteriors(
            embedding_net=embedding_net,
            posteriors_config=config["model"]["neural_posteriors"],
        )
        train_args = config["train_args"]
        output_path = Path(config["output_path"])
        return cls(
            prior=prior,
            inference_class=inference_class,
            neural_posteriors=neural_posteriors,
            device=config["device"],
            embedding_net=embedding_net,
            train_args=train_args,
            output_path=output_path,
        )


    @classmethod
    def load_neural_posteriors(
        cls,
        embedding_net: nn.Module,
        posteriors_config: List[Dict],
    ) -> List[Callable]:
        """Load the inference model

        Args:
            embedding_net (nn.Module): neural network to compress high dimensional data
            posterior_config(List[Dict]): list with configurations for each neural posterior
            model in the ensemble

        Returns:
            List[Callable]: list of neural posterior models with forward methods
        """
        neural_posteriors = []
        for model_args in posteriors_config:
            neural_posteriors.append(
                sbi.utils.posterior_nn(
                    embedding_net=embedding_net,
                    **model_args,
                )
            )
        return neural_posteriors


    def __call__(self, loader):
        """Train your posterior and save it to file

        Args:
            loader (BaseLoader): data loader with stored summary-parameter pairs
        """

        t0 = time.time()
        x = torch.Tensor(loader.get_all_data())
        theta = torch.Tensor(loader.get_all_parameters())
        posteriors, val_loss = [], []
        for n, posterior in enumerate(self.neural_posteriors):
            logging.info(
                f"Training model {n+1} out of {len(self.neural_posteriors)} ensemble models"
            )
            model = self.inference_class(
                prior=self.prior,
                density_estimator=posterior,
                device=self.device,
            )
            model = model.append_simulations(theta, x)
            if not isinstance(self.embedding_net, nn.Identity):
                self.embedding_net.initalize_model(n_input=x.shape[-1])
            density_estimator = model.train(
                **self.train_args,
            )
            posteriors.append(model.build_posterior(density_estimator))
            val_loss += model.summary["best_validation_log_prob"]
        posterior = NeuralPosteriorEnsemble(
            posteriors=posteriors,
            weights=torch.tensor([float(vl) for vl in val_loss]),
        )
        with open(self.output_path / "posterior.pkl", "wb") as handle:
            pickle.dump(posterior, handle)
        logging.info(f"It took {time.time() - t0} seconds to train all models.")


class SBIRunnerSequential(SBIRunner):
    """
    Class to train posterior inference models using the sbi package with multiple rounds
    """

    def __call__(self, loader):
        """Train your posterior and save it to file

        Args:
            loader (BaseLoader): data loader with ability to simulate summary-parameter pairs

        """

        t0 = time.time()
        x_obs= loader.get_obs_data()
        
        all_model = []
        for n, posterior in enumerate(self.neural_posteriors):
            all_model.append(self.inference_class(
                    prior=self.prior,
                    density_estimator=posterior,
                    device=self.device,
                ))
        proposal = self.prior

        for rnd in range(self.num_rounds):
            t1 = time.time()
            logging.info(
                f"Running round {rnd+1} of {self.num_rounds}"
            )
            theta, x = loader.simulate(proposal)
            theta, x = torch.Tensor(theta), torch.Tensor(x)
            posteriors, val_loss = [], []
            for i in range(len(self.neural_posteriors)):
                logging.info(
                    f"Training model {n+1} out of {len(self.neural_posteriors)} ensemble models"
                )
                if not isinstance(self.embedding_net, nn.Identity):
                    self.embedding_net.initalize_model(n_input=x.shape[-1])
                density_estimator = all_model[i].append_simulations(theta, x, proposal).train(
                        **self.train_args,
                )
                posteriors.append(all_model[i].build_posterior(density_estimator))
                val_loss.append(all_model[i].summary["best_validation_log_prob"][-1])
            
            val_loss = torch.tensor([float(vl) for vl in val_loss])
            # Subtract maximum loss to improve numerical stability of exp (cancels in next line)
            val_loss = torch.exp(val_loss - val_loss.max())
            val_loss /= val_loss.sum()
            
            posterior = NeuralPosteriorEnsemble(
                posteriors=posteriors,
                weights=val_loss
            )
            
            with open(self.output_path / f"posterior_{rnd}.pkl", "wb") as handle:
                pickle.dump(posterior, handle)
            proposal = posterior.set_default_x(x_obs)
            logging.info(f"It took {time.time() - t1} seconds to complete round {rnd+1}.")

        with open(self.output_path / "posterior.pkl", "wb") as handle:
            pickle.dump(posterior, handle)
        logging.info(f"It took {time.time() - t0} seconds to train all models.")

