import numpy as np
from pydelfi.delfi import Delfi


class DelfiWrapper(Delfi):
    """Wrapper for pydelfi.delfi.Delfi which unifies sampling interface with sbi engines.
    """
    
    def sample(
        self,
        sample_shape, 
        x, 
        show_progress_bars = None
    ):
        flat_sample_shape = np.prod(sample_shape)
        
        log_target = lambda t: DelfiEnsemble.log_posterior_stacked(t, x[0])
        chain, _, _ = DelfiEnsemble.emcee_sample(log_target, main_chain=flat_sample_shape)
        
        return chain.reshape((*sample_shape, chain.shape[-1]))
    