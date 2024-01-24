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
    """

    def __init__(
        self, n_hidden: List[int], act_fn: str = "SiLU"
    ):
        super().__init__()
        self.act_fn = getattr(nn, act_fn)()
        self.n_layers = len(n_hidden)
        self.n_hidden = n_hidden

    def initalize_model(self, n_input: int):
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
        return self.mlp(x)
