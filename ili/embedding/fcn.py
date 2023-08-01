"""
Module providing compression networks for summary statistics.
"""

from typing import List
import torch
import torch.nn as nn
try:
    from typing import OrderedDict
except:
    pass


class FCN(nn.Module):
    """Fully connected network to compress a summary statistsic.

    Args:
        n_summary (int): dimensionality of the summary
        n_hidden (List[int]): number of hidden units per layer
        act_fn (str):  activation function to use
    """

    def __init__(
        self, n_summary: int, n_hidden: List[int], act_fn: str = "SiLU"
    ):
        super().__init__()
        self.act_fn = getattr(nn, act_fn)()
        self.n_layers = len(n_hidden)
        self.n_hidden = n_hidden
        self.n_summary = n_summary

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
        model.append((f"mlp{layer+1}", nn.Linear(n_left, self.n_summary)))
        self.mlp = nn.Sequential(OrderedDict(model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network, returns the compressed summary.

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: summary
        """
        return self.mlp(x)
