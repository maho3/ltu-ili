from typing import List
import torch
import torch.nn as nn
from typing import OrderedDict


class CNN1DEmbedding(nn.Module):
    """1D CNN network with fully connected layers to compress data.

    Args:
        n_filters (List[int]): number of filters per convolutional layer
        kernel_sizes (List[int]): kernel sizes per convolutional layer
        strides (List[int]): stride values per convolutional layer
        pool_sizes (List[int]): pooling sizes per convolutional layer
        fc_hidden (List[int]): number of hidden units per fully connected layer
        act_fn (str): activation function to use
    """

    def __init__(
        self, n_filters: List[int], kernel_sizes: List[int],
        strides: List[int], pool_sizes: List[int],
        fc_hidden: List[int], act_fn: str = "SiLU"
    ):
        super().__init__()
        self.act_fn = getattr(nn, act_fn)()
        self.n_layers = len(n_filters) + len(pool_sizes)
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.pool_sizes = pool_sizes
        self.fc_hidden = fc_hidden

    def initialize_model(self, n_input: int):
        """Initialize network once the input dimensionality is known.

        Args:
            n_input (int): input dimensionality
        """
        # Convolutional layers
        conv_model = []
        n_left = n_input
        layer, iconv, ipool = 0, 0, 0
        while layer < self.n_layers:
            if layer % 3 == 2:
                pass
                conv_model.append((f"pool{ipool}", nn.MaxPool1d(
                    kernel_size=self.pool_sizes[ipool],
                    stride=self.pool_sizes[ipool]
                )))
                n_left = n_left // self.pool_sizes[ipool]
                ipool += 1
            else:
                conv_model.append((f"conv{iconv}", nn.Conv1d(
                    in_channels=1 if layer == 0 else self.n_filters[iconv - 1],
                    out_channels=self.n_filters[iconv],
                    kernel_size=self.kernel_sizes[iconv],
                    stride=self.strides[iconv],
                    padding=self.kernel_sizes[iconv]//2
                )))
                conv_model.append((f"act{iconv}", self.act_fn))
                n_left = (n_left - self.kernel_sizes[iconv] +
                          self.kernel_sizes[iconv]) // self.strides[iconv]
                iconv += 1
            layer += 1

        self.conv = nn.Sequential(OrderedDict(conv_model))

        # Fully connected layers
        fc_model = []
        fc_input_size = self.n_filters[-1] * n_left
        for layer, hidden_units in enumerate(self.fc_hidden):
            fc_model.append((f"fc{layer}", nn.Linear(
                fc_input_size if layer == 0 else self.fc_hidden[layer - 1],
                hidden_units
            )))
            fc_model.append((f"act_fc{layer}", self.act_fn))

        self.fc = nn.Sequential(OrderedDict(fc_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network, returns the compressed data
        vector.

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: data
        """
        x = x.unsqueeze(1)  # Add a channel dimension (1D convolution)
        x = self.conv(x).squeeze(-1)
        x = self.fc(x)
        return x
