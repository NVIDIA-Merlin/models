import re
from inspect import getfullargspec
from typing import List, Sequence, Type, TypeVar, Union

import torch.nn as nn

from merlin.models.utils.misc_utils import filter_kwargs


class ModuleHook(nn.Module):
    """A helper module to be able to execute Module as a (pre) forward-hook."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, parent, inputs, outputs=None):
        """
        Forward pass for the ModuleHook.

        This function is called when the hook is executed during the
        forward pass of the parent module.

        Args:
            parent (nn.Module): The parent module for which the hook is registered.
            inputs: The inputs to the parent module.
            outputs (optional): The outputs of the parent module (only provided for post-hooks).

        Returns:
            The result of executing the hook module with the given inputs or outputs.
        """
        del parent

        x = inputs if outputs is None else outputs
        if isinstance(x, tuple):
            return self.module(*x)

        return self.module(x)


def has_custom_call(module: nn.Module) -> bool:
    module_call = getattr(module, "__call__", None)
    base_call = getattr(nn.Module, "__call__", None)

    if module_call is None or base_call is None:
        return False

    return module_call.__func__ != base_call


def apply(
    module: Union[nn.Module, Sequence[nn.Module]], inputs, *args, model_context=None, **kwargs
):
    """
    Calls a module with the given inputs and filters kwargs. Returns the output.

    Parameters
    ----------
    module : torch.nn.Module or List[torch.nn.Module]
        The PyTorch module or list of modules to call.
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

    if isinstance(module, (list, tuple, nn.ModuleList)):
        output = inputs
        for i, mod in enumerate(module):
            if i == 0:
                output = apply(mod, output, *args, model_context=model_context, **kwargs)
            else:
                output = apply(mod, output, model_context=model_context)

        return output

    filtered_kwargs = filter_kwargs(kwargs, module, **_k)

    if not has_custom_call(module) or getattr(module, "check_forward", False):
        # We need to check the forward method on the type since when the model gets saved
        # we can't infer the kwargs from using `module.forward` directly
        forward_fn = type(module).forward
        filtered_kwargs = filter_kwargs(filtered_kwargs, forward_fn, **_k)

    return module(inputs, *args, **filtered_kwargs)


def get_all_children(module: nn.Module) -> List[nn.Module]:
    children = []
    for child in module.children():
        if not isinstance(child, nn.ModuleList):
            children.append(child)
        children.extend(get_all_children(child))

    return children


ToSearch = TypeVar("ToSearch", bound=Type[nn.Module])


def find_all_instances(module: nn.Module, to_search: ToSearch) -> List[ToSearch]:
    if isinstance(to_search, nn.Module):
        to_search = type(to_search)

    if isinstance(module, to_search):
        return [module]
    elif module == to_search:
        return [module]

    children = module.children()
    if children:
        results = []
        for sub_module in children:
            results.extend(find_all_instances(sub_module, to_search))

        return results

    return []


def _to_snake_case(name):
    intermediate = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    insecure = re.sub("([a-z])([A-Z])", r"\1_\2", intermediate).lower()
    # If the class is private the name starts with "_" which is not secure
    # for creating scopes. We prefix the name with "private" in this case.
    if insecure[0] != "_":
        return insecure
    return "private" + insecure


def module_name(module: nn.Module, snakecase=True) -> str:
    cls_name = module.__class__.__name__

    if snakecase:
        return _to_snake_case(cls_name)

    return cls_name
