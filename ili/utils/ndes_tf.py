"""
Module to provide loading functions for ndes in various backends.

All Mixture Density Networks (mdn) have the configuration:
    hidden_features (int): width of hidden layers (each MDN has 3 hidden layers)
    num_components (int): number of Gaussian components in the mixture model

All flow-based models (maf, nsf, made) have the configuration:
    hidden_features (int): width of hidden layers in the coupling layers
    num_transforms (int): number of coupling layers

Linear classifiers (linear) have no arguments.

MLP and ResNet classifiers (mlp, resnet) have the configuration:
    hidden_features (int): width of hidden layers (each has 2 hidden layers)
"""

import logging

import pydelfi
import tensorflow as tf


def load_nde_pydelfi(
    n_params: int,
    n_data: int,
    model: str,
    index: int = 0,
    **model_args
):
    """ Load an nde from pydelfi.

    Args:
        n_params (int): dimensionality of parameters
        n_data (int): dimensionality of data points
        model (str): model to use. 
            One of: mdn, maf.
        index (int, optional): index of the nde in the ensemble. Defaults to 0.
        **model_args: additional arguments to pass to the model.
    """
    if model == 'mdn':
        if not (set(model_args.keys()) <= {'hidden_features', 'num_components'}):
            raise ValueError(f"Model {model} arguments mispecified.")
        cfg = {'hidden_features': 50, 'num_components': 1}
        cfg.update(model_args)
        n_hidden = [cfg['hidden_features']] * 3
        activations = [tf.tanh] * 3
        return pydelfi.ndes.MixtureDensityNetwork(
            n_parameters=n_params,
            n_data=n_data,
            n_components=cfg['num_components'],
            n_hidden=n_hidden,
            activations=activations,
            index=index,
        )
    elif model == 'maf':
        if not (set(model_args.keys()) <= {'hidden_features', 'num_transforms'}):
            raise ValueError(f"Model {model} arguments mispecified.")
        cfg = {'hidden_features': 50, 'num_transforms': 4}
        cfg.update(model_args)
        n_hidden = [cfg['hidden_features']] * \
            cfg['num_transforms']
        return pydelfi.ndes.ConditionalMaskedAutoregressiveFlow(
            n_parameters=n_params,
            n_data=n_data,
            n_hiddens=n_hidden,
            n_mades=cfg['num_transforms'],
            act_fun=tf.tanh,
            index=index,
        )
    else:
        raise NotImplementedError(f"Model {model} not implemented for pydelfi")
