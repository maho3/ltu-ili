import pickle
import emcee
import numpy as np
from typing import Dict, Any, List, Callable
from pydelfi.delfi import Delfi
from ili.utils import load_class, load_from_config


class DelfiWrapper(Delfi):
    def __init__(
        self,
        config_ndes: List[Dict],
        **kwargs
    ):
        """Wrapper for pydelfi.delfi.Delfi which adds some necessary functionality and interface.

        Args:
            config_ndes (List[Dict]): list with configurations for each neural posterior
            model in the ensemble

        Other parameters are passed as input to the pydelfi.delfi.Delfi class
        """
        super().__init__(**kwargs)
        self.config_ndes = config_ndes
    
    def sample(
        self,
        sample_shape: tuple,
        x: np.array,
        show_progress_bars=False,
        burn_in_chain=100
    ) -> np.array:
        """Modification of Delfi.emcee_sample designed to conform with the sbi.utils.posterior_ensemble sampler

        Args:
            sample_shape (tuple[int]): size of samples to generate with each MCMC walker, after burn-in
            x (np.array): data vector to condition the inference on
            show_progress_bars (bool): whether to print sampling progress
            burn_in_chain (int): length of burn-in for MCMC sampling

        Returns:
            np.array: array of unique samples of shape (# of samples, # of parameters), after MCMC rejection
        """
        num_samples = np.prod(sample_shape)
        
        # build posterior to sample
        log_target = lambda t: self.log_posterior_stacked(t, x)
        
        # Initialize walkers
        x0 = self.posterior_samples[
             np.random.choice(np.arange(len(self.posterior_samples)),
                              p=self.posterior_weights.astype(np.float32)/sum(self.posterior_weights),
                              replace=False,
                              size=self.nwalkers),
             :]
        
        # Set up the sampler
        sampler = emcee.EnsembleSampler(self.nwalkers, self.npar, log_target)

        # Burn-in chain
        state = sampler.run_mcmc(x0, burn_in_chain, progress=show_progress_bars)
        sampler.reset()

        # Main chain
        sampler.run_mcmc(state, num_samples, progress=show_progress_bars)

        # pull out the unique samples and weights
        chain, weights = np.unique(sampler.get_chain(flat=True), axis=0, return_counts=True)

        # pull out the log probabilities
        log_prob, _ = np.unique(sampler.get_log_prob(flat=True), axis=0, return_counts=True)

        return chain  # , weights, log_prob
    
    
    @classmethod
    def load_ndes(
        cls,
        config_ndes: List[Dict],
        n_params: int,
        n_data: int,
    ) -> List[Callable]:
        """Load the inference model

        Args:
            config_ndes(List[Dict]): list with configurations for each neural posterior
            model in the ensemble
            n_params (int): dimensionality of each parameter vector
            n_data (int): dimensionality of each datapoint

        Returns:
            List[Callable]: list of neural posterior models with forward methods
        """
        neural_posteriors = []
        for i, model_args in enumerate(config_ndes):
            model_args['args']['index'] = i
            model_args['args']['n_parameters'] = n_params
            model_args['args']['n_data'] = n_data
            # layer activations must be input as TF classes
            if 'act_fun' in model_args['args']:
                if isinstance(model_args['args']['act_fun'], str):
                    model_args['args']['act_fun'] = load_class('tensorflow', model_args['args']['act_fun'])
            elif 'activations' in model_args['args']:
                model_args['args']['activations'] = [load_class('tensorflow', x) if isinstance(x, str) else x for x in model_args['args']['activations']]
            
            neural_posteriors.append(
                load_from_config(model_args)
            )
        return neural_posteriors
    
    
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
            'data': self.data,
            'prior': self.prior,
            'config_ndes': self.config_ndes,
            'results_dir': self.results_dir,
            'param_names': self.names,
            # this hack is required because Delfi overrides these parameters
            'graph_restore_filename': self.graph_restore_filename.split('/')[-1],
            'restore_filename': self.restore_filename.split('/')[-1]
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
        metadata.pop('n_params')
        metadata.pop('n_data')
            
        return cls(**metadata, nde=ndes, restore=True)
    
    