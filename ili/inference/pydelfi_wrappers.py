import pickle
import emcee
import numpy as np
from typing import Dict, Any, List, Callable
from pydelfi.delfi import Delfi
from ili.utils import load_class, load_from_config


class DelfiWrapper(Delfi):
    """Wrapper for pydelfi.delfi.Delfi which adds necessary functionality and interface.
    """
    def __init__(self, config_ndes, **kwargs):
        super().__init__(**kwargs)
        self.config_ndes = config_ndes
    
    def sample(
        self,
        sample_shape, 
        x,
        show_progress_bars = False,
        burn_in_chain = 100
    ):
        """Modification of Delfi.emcee_sample
        """
        # build posterior to sample
        log_target = lambda t: self.log_posterior_stacked(t, x)
        
        # Initialize walkers
        x0 = self.posterior_samples[np.random.choice(np.arange(len(self.posterior_samples)), p=self.posterior_weights.astype(np.float32)/sum(self.posterior_weights), replace=False, size=self.nwalkers),:]
        
        # Set up the sampler
        sampler = emcee.EnsembleSampler(self.nwalkers, self.npar, log_target)

        # Burn-in chain
        state = sampler.run_mcmc(x0, burn_in_chain, progress=show_progress_bars)
        sampler.reset()

        # Main chain
        sampler.run_mcmc(state, sample_shape, progress=show_progress_bars)

        # pull out the unique samples and weights
        chain, weights = np.unique(sampler.get_chain(flat=True), axis=0, return_counts=True)

        # pull out the log probabilities
        log_prob, _ = np.unique(sampler.get_log_prob(flat=True), axis=0, return_counts=True)

        return chain # , weights, log_prob
    
    
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
        meta_filename,
    ):
        
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
        meta_path
    ):
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
            
        ndes = cls.load_ndes(
            n_params = metadata['n_params'],
            n_data = metadata['n_data'],
            config_ndes = metadata['config_ndes']
        )
        metadata.pop('n_params')
        metadata.pop('n_data')
            
        return cls(**metadata, nde=ndes, restore=True)
    
    