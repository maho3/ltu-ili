"""
Module providing CNN compression networks for data.
"""

from typing import List
import torch
import torch.nn as nn
import numpy as np


class CNN(nn.Module):
    """Convolutional network to compress data.

    Args:
        in_shape (List[int]): shape of the input data
        n_channels (List[int]): number of channels per layer
        L_kernel (List[int]): kernel size per layer
        strides (List[int]): stride per layer
        pool_sizes (List[int]): pooling size per layer
        n_hiddens (List[int]): number of hiddens units per layer
    """

    def __init__(
        self, in_shape: List[int], n_channels: List[int], kernel_sizes: List[int],
        strides: List[int], pool_sizes: List[int], n_hiddens: List[int]
    ):
        super().__init__()
        self.n_channels = [in_shape[0]] + n_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.pool_sizes = pool_sizes
        self.n_hiddens = n_hiddens
        self.in_shape = in_shape

        # initialize the convolutional layers
        self.convs = [
            nn.Conv2d(
                self.n_channels[i], self.n_channels[i+1],
                kernel_size=self.kernel_sizes[i], stride=self.strides[i]
            ) for i in range(len(self.n_channels)-1)
        ]
        self.convs = nn.ModuleList(self.convs)

        # initialize the pooling layers
        self.pools = [
            nn.MaxPool2d(kernel_size=self.pool_sizes[i])
            for i in range(len(self.pool_sizes))
        ]
        self.pools = nn.ModuleList(self.pools)

        # initialize the fully connected layers
        self.act_fn = nn.ReLU()

        # pass a dummy input through the network to get the output shape
        x_ = torch.zeros((1, *self.in_shape))
        x_ = self.conv_forward(x_)
        n_out = np.prod(x_.shape)

        # initialize the fully connected layers
        self.n_hiddens = [n_out] + self.n_hiddens
        self.linears = [
            nn.Linear(self.n_hiddens[i], self.n_hiddens[i+1])
            for i in range(len(self.n_hiddens)-1)
        ]
        self.linears = nn.ModuleList(self.linears)

    def conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the convolutional layers.

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: output
        """
        for i in range(len(self.convs)//2):
            x = self.act_fn(self.convs[2*i](x))
            x = self.act_fn(self.convs[2*i+1](x))
            x = self.pools[i](x)
        return x

    def mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the fully connected layers.

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: output
        """
        for i in range(len(self.linears)-1):
            x = self.act_fn(self.linears[i](x))
        return self.linears[-1](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network, returns the compressed data
        vector.

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: data
        """
        x = self.conv_forward(x)
        x = x.view(x.size(0), -1)
        x = self.mlp_forward(x)
        return x
