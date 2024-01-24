
# Let's build a model similar to your previous one

import torch
from typing import List, Optional
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch import Tensor

from ili.embedding.fcn import FCN

def get_mlp(in_channels, out_channels, hidden_layers,):
    fcn = FCN(
        n_data =out_channels,
        n_hidden = hidden_layers
    )
    fcn.initalize_model(in_channels)
    return fcn

class EdgeUpdate(torch.nn.Module):
    def __init__(
        self,
        edge_in_channels: int,
        edge_out_channels: int,
        hidden_layers: List[int],
    ):
        """Update edge attributes

        Args:
            edge_in_channels (int): input channels
            edge_out_channels (int): output channels of MLP, determines the dimensionality of the messages
            hidden_layers (List[int]): hidden layers of MLP
        """
        super().__init__()
        self.mlp = get_mlp(
            in_channels=edge_in_channels,
            out_channels=edge_out_channels,
            hidden_layers=hidden_layers,
        )

    def forward(self, h_i: Tensor, h_j: Tensor, edge_attr: Tensor, u: Tensor) -> Tensor:
        """

        Args:
            h_i (Tensor): node features node i
            h_j (Tensor): node features node j
            edge_attr (Tensor): edge attributes
            u (Tensor): global attributes

        Returns:
            Tensor: updated edge attributes
        """
        inputs_to_concat = []
        if h_i is not None:
            inputs_to_concat.append(h_i)
        if h_j is not None:
            inputs_to_concat.append(h_j)
        if edge_attr is not None:
            inputs_to_concat.append(edge_attr)
        if u is not None:
            inputs_to_concat.append(u)
        inputs = torch.concat(inputs_to_concat, dim=-1)
        return self.mlp(inputs)

# The node update would do all the work
class NodeUpdate(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_layers: List[int],
        aggr: str = "add",
    ):
        """Update nodes

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            hidden_layers (List[int]): hidden layers of MLP
            aggr (str, optional): Node aggregation. Defaults to 'add'.
        """
        # aggregator use in propagate
        super().__init__(aggr=aggr)
        self.mlp = get_mlp(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_layers=hidden_layers,
        )

    def forward(
        self, h: Tensor, edge_index: Tensor, edge_attr: Tensor, u: Tensor
    ) -> Tensor:
        """Update node features

        Args:
            h (Tensor): node features
            edge_index (Tensor): edge index
            edge_attr (Tensor): edge attributes
            u (Tensor): globals

        Returns:
            Tensor: updated node features
        """
        msg = self.propagate(
            edge_index,
            edge_attr=edge_attr,
        )
        to_concat = [msg]
        if h is not None:
            to_concat.append(h)
        if u is not None:
            to_concat.append(u)
        input = torch.concat(to_concat, dim=-1)
        return self.mlp(input)

    def message(
        self,
        edge_attr: Tensor,
    ) -> Tensor:
        """Message function

        Args:
            edge_attr (Tensor): edge attributes

        Returns:
            Tensor: messages
        """
        return edge_attr

    
# GNN Layer
class GraphLayer(torch.nn.Module):
    def __init__(
        self,
        node_in_channels: int=2,
        node_out_channels: int=1,
        edge_in_channels: int=2,
        hidden_layers: List[int]=[128,128,128],
        edge_out_channels: int=16,
        global_in_channels: int=0,
    ):
        super().__init__()
        node_in_channels = node_in_channels if node_in_channels is not None else 0
        self.edge_update = EdgeUpdate(
            edge_in_channels=node_in_channels * 2
            + edge_in_channels
            + global_in_channels,
            edge_out_channels=edge_out_channels,
            hidden_layers=hidden_layers,
        )
        self.node_update = NodeUpdate(
            node_in_channels + edge_out_channels,
            node_out_channels,
            hidden_layers=hidden_layers,
        )

    def forward(self, h, edge_index, edge_attr, batch=None,):
        row, col = edge_index
        # TODO: properly account for global u
        if h is not None:
            edge_attr = self.edge_update(h[row], h[col], edge_attr, None)
        else:
            edge_attr = self.edge_update(None, None, edge_attr, None)
        return self.node_update(h, edge_index, edge_attr,None), edge_attr

class GraphNetwork(torch.nn.Module):
    def __init__(
            self,
            node_features_dim: Optional[int] = None,
            edge_features_dim: Optional[int] = 3,
            node_features_hidden_dim: int = 32,
            edge_features_hidden_dim: int = 32,
            global_output_dim: int = 16,
            message_passing_steps: int = 3,
            hidden_layers: Optional[List[int]] = [128, 128, 128],
    ):
        super().__init__()
        self.graph_layers = torch.nn.ModuleList() 
        for idx in range(message_passing_steps):
            if idx == 0:
                node_in_channels = node_features_dim
                node_out_channels = node_features_hidden_dim
                edge_in_channels = edge_features_dim
                edge_out_channels = edge_features_hidden_dim
            else:
                node_in_channels = node_features_hidden_dim
                node_out_channels = node_features_hidden_dim
                edge_in_channels = edge_features_hidden_dim
                edge_out_channels = edge_features_hidden_dim
            self.graph_layers.append(
                GraphLayer(
                    node_in_channels=node_in_channels,
                    node_out_channels=node_out_channels,
                    edge_in_channels=edge_in_channels,
                    edge_out_channels=edge_out_channels,
                    hidden_layers=hidden_layers,
                )
            )
        self.global_mlp = get_mlp(node_features_hidden_dim, global_output_dim, hidden_layers=hidden_layers)

    def forward(self, data):
        h, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for layer in self.graph_layers:
            h, edge_attr = layer(h, edge_index, edge_attr, batch)
        h = global_mean_pool(h, batch)  # Global pooling
        return self.global_mlp(h)