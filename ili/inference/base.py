
from typing import Dict, Optional, Any, Union
from pathlib import Path

try:
    from sbi.inference.posteriors.base_posterior import NeuralPosterior
    ModelClass = NeuralPosterior
    interface = 'torch'
except ModuleNotFoundError:
    from ili.inference.pydelfi_wrappers import DelfiWrapper
    ModelClass = DelfiWrapper
    interface = 'tensorflow'


class _BaseRunner():
    def __init__(
        self,
        prior: Any,
        inference_class: ModelClass,
        train_args: Dict = {},
        out_dir: Union[str, Path] = None,
        device: str = 'cpu',
        name: Optional[str] = "",
    ):
        self.prior = prior
        self.inference_class = inference_class
        self.class_name = inference_class.__name__
        self.train_args = train_args
        self.device = device
        self.name = name
        self.out_dir = out_dir
        if self.out_dir is not None:
            self.out_dir = Path(self.out_dir)
            self.out_dir.mkdir(parents=True, exist_ok=True)
