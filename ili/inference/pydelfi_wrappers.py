"""
Module providing wrappers for the pydelfi package to conform with the sbi
interface.
"""

import pickle
import emcee
import numpy as np
from typing import Dict, List, Callable, Optional
from pydelfi.delfi import Delfi
from ili.utils import load_class, load_from_config


class DelfiWrapper(Delfi):
    """Wrapper for pydelfi.delfi.Delfi which adds some necessary
    functionality and interface.

    Args:
        config_ndes (List[Dict]): list with configurations for each neural
        posterior model in the ensemble

    Other parameters are passed as input to the pydelfi.delfi.Delfi class
    """

    def __init__(
        self,
        config_ndes: List[Dict],
        name: Optional[str] = '',
        **kwargs
    ):
        super().__init__(**kwargs)
        kwargs.pop('nde')
        self.kwargs = kwargs
        self.config_ndes = config_ndes
        self.num_components = len(config_ndes)
        self.name = name
        self.prior.sample = self.prior.draw  # aliasing for consistency

    def potential(self, theta: np.array, x: np.array):
        """Modification of Delfi.log_prob designed to conform with the
        form of sbi.utils.posterior_ensemble

        Args:
            theta (np.array): parameter vector
            x (np.array): data vector to condition the inference on

        Returns:
            float: log posterior probability
        """
        return self.log_posterior_stacked(theta, x)

    def sample(
        self,
        sample_shape: tuple,
        x: np.array,
        show_progress_bars=False,
        burn_in_chain=1000
    ) -> np.array:
        """Modification of Delfi.emcee_sample designed to conform with the
        form of sbi.utils.posterior_ensemble

        Args:
            sample_shape (tuple[int]): size of samples to generate with each
                MCMC walker, after burn-in
            x (np.array): data vector to condition the inference on
            show_progress_bars (bool): whether to print sampling progress
            burn_in_chain (int): length of burn-in for MCMC sampling

        Returns:
            np.array: array of unique samples of shape (# of samples, # of
                parameters), after MCMC rejection
        """
        num_samples = np.prod(sample_shape)

        # build posterior to sample
        def log_target(t): return self.log_posterior_stacked(t, x)

        # Initialize walkers
        x0 = self.posterior_samples[
            np.random.choice(np.arange(len(self.posterior_samples)),
                             p=self.posterior_weights.astype(
                                 np.float32)/sum(self.posterior_weights),
                             replace=False,
                             size=self.nwalkers),
            :]

        # Set up the sampler
        sampler = emcee.EnsembleSampler(self.nwalkers, self.npar, log_target)

        # Burn-in chain
        state = sampler.run_mcmc(
            x0, burn_in_chain, progress=show_progress_bars)
        sampler.reset()

        # Main chain
        sampler.run_mcmc(state, num_samples, progress=show_progress_bars)

        # pull out the unique samples and weights
        chain = sampler.get_chain(flat=True)

        return chain

    @classmethod
    def load_ndes(
        cls,
        config_ndes: List[Dict],
        n_params: int,
        n_data: int,
    ) -> List[Callable]:
        """Load the inference model

        Args:
            config_ndes(List[Dict]): list with configurations for each neural
                posterior model in the ensemble
            n_params (int): dimensionality of each parameter vector
            n_data (int): dimensionality of each datapoint

        Returns:
            List[Callable]: list of neural posterior models with forward
                methods
        """
        nets = []
        for i, model_args in enumerate(config_ndes):
            model_args['args']['index'] = i
            model_args['args']['n_parameters'] = n_params
            model_args['args']['n_data'] = n_data
            # layer activations must be input as TF classes
            if 'act_fun' in model_args['args']:
                if isinstance(model_args['args']['act_fun'], str):
                    model_args['args']['act_fun'] = load_class(
                        'tensorflow', model_args['args']['act_fun'])
            elif 'activations' in model_args['args']:
                model_args['args']['activations'] = \
                    [load_class('tensorflow', x)
                     if isinstance(x, str) else x
                     for x in model_args['args']['activations']]

            nets.append(
                load_from_config(model_args)
            )
        return nets

    def save_engine(
        self,
        meta_filename: str,
    ):
        """Save necessary metadata for reloading to file

        Args:
            meta_filename (str): filename of saved metadata
        """
        metadata = {
            'n_data': self.D,
            'n_params': self.npar,
            'name': self.name,
            'config_ndes': self.config_ndes,
            'kwargs': self.kwargs
        }
        with open(self.results_dir + meta_filename, 'wb') as f:
            pickle.dump(metadata, f)

    @classmethod
    def load_engine(
        cls,
        meta_path: str,
    ):
        """Load a DelfiWrapper from metadata file

        Args:
            meta_path (str): path to saved metadata

        Returns:
            DelfiWrapper: a full Delfi inference model with pre-trained weights
        """
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)

        ndes = cls.load_ndes(
            n_params=metadata['n_params'],
            n_data=metadata['n_data'],
            config_ndes=metadata['config_ndes']
        )
        if 'restore' in metadata['kwargs']:
            metadata['kwargs'].pop('restore')

        return cls(
            **metadata['kwargs'],
            nde=ndes,
            config_ndes=metadata['config_ndes'],
            name=metadata['name'],
            restore=True
        )
