from typing import List
import torch
import torch.nn as nn
from typing import OrderedDict


class FCN(nn.Module):
    def __init__(
        self, n_summary: int, n_input: int, n_hidden: List[int], act_fn: str = "SiLU"
    ):
        """fully connected network to compress a summary statistsic

        Args:
            n_summary (int): dimensionality of the summary
            n_input (int): dimensionality of the input
            n_hidden (List[int]): number of hidden units per layer
            act_fn (str):  activation function to use
        """
        super().__init__()
        act_fn = getattr(nn, act_fn)()
        n_layers = len(n_hidden)
        model = []
        for layer in range(n_layers):
            n_left = n_hidden if layer > 0 else n_input
            model.append((f"mlp{layer}", nn.Linear(n_left, n_hidden)))
            model.append((f"act{layer}", act_fn))
        model.append((f"mlp{layer+1}", nn.Linear(n_hidden, n_summary)))
        self.mlp = nn.Sequential(OrderedDict(model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass of the neural network, returns the summary

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: summary
        """
        return self.mlp(x)
