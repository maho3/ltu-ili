"""
Module providing compression networks for data.
"""

from typing import List
import torch
import torch.nn as nn
from typing import OrderedDict


class FCN(nn.Module):
    """Fully connected network to compress data.

    Args:
        n_hidden (List[int]): number of hidden units per layer
        act_fn (str):  activation function to use
        n_input (int): dimensionality of the input (optional)
    """

    def __init__(
        self, n_hidden: List[int], act_fn: str = "SiLU", n_input=None
    ):
        super(FCN, self).__init__()
        self.act_fn = getattr(nn, act_fn)()
        self.n_layers = len(n_hidden)
        self.n_hidden = n_hidden

        # allows to have non empty Parameters for check_net_device in sbi
        self.dummy = nn.Parameter(torch.Tensor([0]), requires_grad=False)

        # Allows to specify n_input
        if n_input is not None:
            self.initialize_model(n_input)

    def initialize_model(self, n_input: int):
        """Initialize network once the input dimensionality is known.

        Args:
            n_input (int): input dimensionality
        """
        model = []
        n_left = n_input
        for layer in range(self.n_layers):
            model.append((f"mlp{layer}", nn.Linear(
                n_left, self.n_hidden[layer])))
            model.append((f"act{layer}", self.act_fn))
            n_left = self.n_hidden[layer]
        model.pop()  # remove last activation
        self.mlp = nn.Sequential(OrderedDict(model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network, returns the compressed data
        vector.

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: data
        """
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        if not hasattr(self, "mlp"):
            self.initialize_model(x.shape[-1])

        return self.mlp(x)
