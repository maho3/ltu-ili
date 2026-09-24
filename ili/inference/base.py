
from pathlib import Path
from typing import Any

try:
    from sbi.inference.posteriors.base_posterior import NeuralPosterior
    ModelClass = NeuralPosterior
    interface = 'torch'
except ModuleNotFoundError:
    from ili.inference.pydelfi_wrappers import DelfiWrapper
    ModelClass = DelfiWrapper
    interface = 'tensorflow'


class _BaseRunner:
    def __init__(
        self,
        prior: Any,
        train_args: dict | None = None,
        out_dir: str | Path | None = None,
        device: str = 'cpu',
        name: str | None = "",
    ):
        self.prior = prior
        if train_args is None:
            train_args = {}
        self.train_args = train_args
        self.device = device
        self.name = name
        self.out_dir = out_dir
        if self.out_dir is not None:
            self.out_dir = Path(self.out_dir)
            self.out_dir.mkdir(parents=True, exist_ok=True)
