"""
Module containing a pytorch implementation of EvidenceNetworks, specifically
the tutorial model in:
https://github.com/NiallJeffrey/EvidenceNetworksDemo/blob/main/network_architecture.py

All usage of this model should cite Jeffrey & Wandelt (2024) - arxiv:2305.11241

TODO:
  *  Add embedding network to the model.
"""

import torch
import torch.nn as nn


# def smooth_sign(x, k=100.):
#     return 2. * torch.sigmoid(k * x) - 1


def parity_odd_power(x, alpha=2):
    return x * (torch.abs(x) ** (alpha - 1))


def leaky_parity_odd_power(x, alpha=2):
    return x + parity_odd_power(x, alpha)


class POPExpLoss(nn.Module):
    def forward(self, model_pred, model_label):
        model_pred = leaky_parity_odd_power(model_pred, alpha=1)
        model_pred = torch.clamp(model_pred, -50, 50)
        loss_val = torch.exp((0.5 - model_label) * model_pred)
        return torch.mean(loss_val)


class ExpLoss(nn.Module):
    def forward(self, model_pred, model_label):
        model_pred = torch.clamp(model_pred, -50, 50)
        loss_val = torch.exp((0.5 - model_label) * model_pred)
        return torch.mean(loss_val)


class EvidenceNetworkSimple(nn.Module):
    def __init__(
        self,
        input_size,
        layer_width=16,
        added_layers=3,
        batch_norm_flag=1,
        alpha=2
    ):
        super().__init__()
        self.input_size = input_size
        self.layer_width = layer_width
        self.added_layers = added_layers
        self.batch_norm_flag = batch_norm_flag
        self.alpha = alpha

        self.initial_layer = self.simple_layer(
            self.input_size, self.layer_width)

        self.hidden_layers = nn.ModuleList(
            [self.simple_layer(self.layer_width, self.layer_width)
             for _ in range(2)])

        self.residual_layers = nn.ModuleList(
            [self.residual_layer(self.layer_width, self.layer_width)
             for _ in range(self.added_layers)])

        self.output_layer = nn.Linear(self.layer_width, 1)

    def simple_layer(self, in_features, out_features):
        layers = [
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(0.1)
        ]
        if self.batch_norm_flag == 1:
            layers.append(nn.BatchNorm1d(self.layer_width))
        return nn.Sequential(*layers)

    def residual_layer(self, in_features, out_features):
        layers = [
            self.simple_layer(in_features, out_features),
            self.simple_layer(out_features, out_features)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_layer(x)

        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)

        # Residual layers
        for layer in self.residual_layers:
            x = x + layer(x)

        # Output layer
        x = self.output_layer(x)
        x = 0.1 * x + 0.001
        x = leaky_parity_odd_power(x, alpha=self.alpha)
        return x
