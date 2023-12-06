
try:
    import torch
    from sbi.utils import BoxUniform as ModuleUniform
    backend = 'sbi'
except ModuleNotFoundError:
    from pydelfi.priors import Uniform as ModuleUniform
    backend = 'pydelfi'


class Uniform(ModuleUniform):
    """Wrapper for sbi and pydelfi uniform distributions
    which allows to use the same code for both backends.
    Takes the lower and upper bounds of the uniform distribution
    over the parameters and provides logpdf and sample methods.

    Args:
        low (vector): lower bound of the uniform distribution
        high (vector): upper bound of the uniform distribution
    """

    def __init__(self, low, high, device='cpu'):
        if backend == 'sbi':
            low = torch.tensor(low).to(device)
            high = torch.tensor(high).to(device)
        super().__init__(low, high)
