from inspect import getfullargspec

import torch.nn as nn

from merlin.models.utils.misc_utils import filter_kwargs


def has_custom_call(module: nn.Module) -> bool:
    module_call = getattr(module, "__call__", None)
    base_call = getattr(nn.Module, "__call__", None)

    if module_call is None or base_call is None:
        return False

    return module_call.__func__ != base_call


def apply_module(module: nn.Module, inputs, *args, **kwargs):
    """
    Calls a module with the given inputs and filters kwargs. Returns the output.

    Parameters
    ----------
    module : torch.nn.Module
        The PyTorch module to call.
    inputs : torch.Tensor
        The input tensor to be passed to the module.
    *args : tuple
        Additional arguments to be passed to the module.
    **kwargs : dict
        Additional keyword arguments to be filtered and passed to the module.

    Returns
    -------
    output : torch.Tensor
        The output tensor after processing the input by the module.
    """

    _k = dict(cascade_kwargs_if_possible=True, argspec_fn=getfullargspec)

    filtered_kwargs = filter_kwargs(kwargs, module, **_k)

    if not has_custom_call(module):
        # We need to check the forward method on the type since when the model gets saved
        # we can't infer the kwargs from using `module.forward` directly
        forward_fn = type(module).forward
        filtered_kwargs = filter_kwargs(filtered_kwargs, forward_fn, **_k)

    return module(inputs, *args, **filtered_kwargs)
