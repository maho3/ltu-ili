"""
Module providing wrappers for the pydelfi package to conform with the sbi
interface.
"""

import pickle
import emcee
import numpy as np
from math import ceil
from typing import Dict, List, Callable, Optional, Union
from pydelfi.delfi import Delfi
from ili.utils import load_class, load_from_config, load_nde_pydelfi


class DelfiWrapper(Delfi):
    """Trainer for a neural posterior ensemble using the pydelfi package.
    Wrapper for pydelfi.delfi.Delfi which adds some necessary
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
        self.name = name
        self.num_components = len(config_ndes)
        self.name = name
        self.prior.sample = self.prior.draw  # aliasing for consistency

    def potential(self, theta: np.array, x: np.array):
        """Returns the log posterior probability of a data vector given
        parameters. Modification of Delfi.log_prob designed to conform
        with the form of sbi.utils.posterior_ensemble

        Args:
            theta (np.array): parameter vector
            x (np.array): data vector to condition the inference on

        Returns:
            float: log posterior probability
        """
        return self.log_posterior_stacked(theta, x)

    def sample(
        self,
        sample_shape: Union[int, tuple],
        x: np.array,
        show_progress_bars=False,
        num_chains: int = 10,
        burn_in=200,
        thin=3,
        skip_initial_state_check: bool = False
    ) -> np.array:
        """Samples from the posterior distribution using MCMC rejection.
        Modification of Delfi.emcee_sample designed to conform with the
        form of sbi.utils.posterior_ensemble

        Args:
            sample_shape (int, tuple[int]): size of samples to generate with
                each MCMC walker, after burn-in
            x (np.array): data vector to condition the inference on
            show_progress_bars (bool): whether to print sampling progress
            num_chains (int): number of MCMC chains to run in parallel
            burn_in (int): length of burn-in for MCMC sampling
            thin (int): thinning factor for MCMC sampling
            skip_initial_state_check (bool): whether to skip the initial state
                check for the MCMC sampler

        Returns:
            np.array: array of unique samples of shape (# of samples, # of
                parameters), after MCMC rejection
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)

        # calculate number of samples per chain
        num_samples = np.prod(sample_shape)
        per_chain = ceil(num_samples / num_chains)

        # build posterior to sample
        def log_target(t, x):
            return self.potential(t, x)

        # Initialize walkers
        theta0 = np.stack([self.prior.sample()
                           for _ in range(num_chains)])

        # Set up the sampler
        sampler = emcee.EnsembleSampler(
            num_chains,
            self.npar,
            log_target,
            vectorize=False,
            args=(x,),
        )

        # Sample
        sampler.run_mcmc(
            theta0,
            burn_in + per_chain,
            thin_by=thin,
            progress=show_progress_bars,
            skip_initial_state_check=skip_initial_state_check
        )

        # Pull out the unique samples and weights
        chain = sampler.get_chain(discard=burn_in, flat=True)[:num_samples]

        return chain.reshape((*sample_shape, self.npar))

    @staticmethod
    def load_ndes(
        config_ndes: List[Dict],
        n_params: int,
        n_data: int,
    ) -> List[Callable]:
        """Initialize the neural density estimators from configuration yamls.

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
            nets.append(
                load_nde_pydelfi(
                    n_params=n_params, n_data=n_data,
                    index=i, **model_args))
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
