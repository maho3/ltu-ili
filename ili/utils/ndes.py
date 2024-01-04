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

try:
    import sbi
    from torch.nn import Module, Identity
    Identity = Identity()
    interface = 'torch'
except ModuleNotFoundError:
    import pydelfi
    import tensorflow as tf
    interface = 'tensorflow'
    Module, Identity = None, None


def load_nde_sbi(
        engine: str,
        model: str,
        embedding_net: Module = Identity,
        **model_args):
    """Load an nde from sbi.

    Args:
        engine (str): engine to use. 
            One of: NPE, NLE, NRE, SNPE, SNLE, or SNRE.
        model (str): model to use. 
            One of: mdn, maf, nsf, made, linear, mlp, resnet.
        embedding_net (nn.Module, optional): embedding network to use.
            Defaults to nn.Identity().
        **model_args: additional arguments to pass to the model.
    """
    # load NRE models (linear, mlp, resnet)
    if 'NRE' in engine:
        if model not in ['linear', 'mlp', 'resnet']:
            raise ValueError(f"Model {model} not implemented for {engine}.")
        return sbi.utils.classifier_nn(
            model=model, embedding_net_x=embedding_net, **model_args)

    if model not in ['mdn', 'maf', 'nsf', 'made']:
        raise ValueError(f"Model {model} not implemented for {engine}.")

    if (model == 'mdn'):
        # check for arguments
        if not (set(model_args.keys()) <= {'hidden_features', 'num_components'}):
            raise ValueError(f"Model {model} arguments mispecified.")
    else:
        # check for arguments
        if not (set(model_args.keys()) <= {'hidden_features', 'num_transforms'}):
            raise ValueError(f"Model {model} arguments mispecified.")

    # Load NPE models (mdn, maf, nsf, made)
    if 'NPE' in engine:
        return sbi.utils.posterior_nn(
            model=model, embedding_net=embedding_net, **model_args)

    # Load NLE models (mdn, maf, nsf, made)
    if 'NLE' in engine:
        if embedding_net != Identity:
            logging.warning(
                "Using an embedding_net with NLE models compresses theta, not "
                "x as might be expected.")
        return sbi.utils.likelihood_nn(
            model=model, embedding_net=embedding_net, **model_args)

    raise ValueError(f"Engine {engine} not implemented.")


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
        if not (set(model_args.keys()) <= {'hidden_features', 'num_transforms'}):
            raise ValueError(f"Model {model} arguments mispecified.")
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
