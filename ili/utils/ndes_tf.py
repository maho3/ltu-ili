"""
Module to provide loading functions for ndes in pydelfi.

All Mixture Density Networks (mdn) have the configuration:
    hidden_features (int): width of hidden layers (each MDN has 3 hidden layers)
    num_components (int): number of Gaussian components in the mixture model

All flow-based models (maf) have the configuration:
    hidden_features (int): width of hidden layers in the coupling layers
    num_transforms (int): number of coupling layers

"""

import pydelfi
import tensorflow as tf
import logging


def load_nde_pydelfi(
    n_params: int,
    n_data: int,
    model: str,
    index: int = 0,
    engine: str = 'NLE',
    **model_args
):
    """ Load an nde from pydelfi.

    Args:
        n_params (int): dimensionality of parameters
        n_data (int): dimensionality of data points
        model (str): model to use. 
            One of: mdn, maf.
        index (int, optional): index of the nde in the ensemble. Defaults to 0.
        engine (str, optional): dummy argument to match sbi interface.
            Must be set to 'NLE' or will be overwritten.
        **model_args: additional arguments to pass to the model.
    """
    if 'NLE' not in engine:
        raise ValueError(
            f'Engine {engine} not supported in pydelfi backend. '
            'You probably meant to specify engine="NLE" or to use the NPE or NRE'
            ' engines in the sbi or lampe backends.')
    model = model.lower()

    # check the model parameterizations
    if model == 'mdn':
        model_defaults = dict(hidden_features=16, num_components=3)
    else:
        model_defaults = dict(hidden_features=16, num_transforms=2)
    if not (set(model_args.keys()) <= set(model_defaults.keys())):
        raise ValueError(
            f"Model {model} arguments mispecified. Extra arguments found: "
            f"{set(model_args.keys()) - set(model_defaults.keys())}.")

    # set defaults
    model_args = {**model_defaults, **model_args}

    # setup models
    if model == 'mdn':
        n_hidden = [model_args['hidden_features']] * 3
        activations = [tf.tanh] * 3
        return pydelfi.ndes.MixtureDensityNetwork(
            n_parameters=n_params,
            n_data=n_data,
            n_components=model_args['num_components'],
            n_hidden=n_hidden,
            activations=activations,
            index=index,
        )
    elif model == 'maf':
        n_hidden = [model_args['hidden_features']] * \
            model_args['num_transforms']
        return pydelfi.ndes.ConditionalMaskedAutoregressiveFlow(
            n_parameters=n_params,
            n_data=n_data,
            n_hiddens=n_hidden,
            n_mades=model_args['num_transforms'],
            act_fun=tf.tanh,
            index=index,
        )
    else:
        raise NotImplementedError(f"Model {model} not implemented for pydelfi")
